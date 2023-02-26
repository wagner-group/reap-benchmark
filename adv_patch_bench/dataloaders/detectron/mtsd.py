"""Register and load MTSD dataset."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm.auto import tqdm

from adv_patch_bench.utils.argparse import parse_dataset_name
from adv_patch_bench.utils.types import DetectronSample
from hparams import (
    DATASET_METADATA,
    DEFAULT_PATH_MTSD_LABEL,
    OLD_TO_NEW_LABELS,
    PATH_DUPLICATE_FILES,
    TS_COLOR_DICT,
    TS_COLOR_OFFSET_DICT,
)

_ALLOWED_SPLITS = ("train", "test", "val")
_NUM_KEYPOINTS = 4


def _readlines(path: str) -> List:
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def get_mtsd_anno(base_path: str, dataset_name: str) -> Dict[str, Any]:
    """Get MTSD annotation and metadata needed for loading dataset."""
    class_names: list[str] = list(
        DATASET_METADATA[dataset_name]["class_name"].values()
    )
    bg_idx = len(class_names) - 1
    label_path: pathlib.Path = pathlib.Path(base_path) / "annotations"
    similarity_df_csv_path: str = PATH_DUPLICATE_FILES
    duplicate_files_df: pd.DataFrame = pd.read_csv(similarity_df_csv_path)

    # Load annotation file that contains dimension and shape of each MTSD label
    label_map: pd.DataFrame = pd.read_csv(DEFAULT_PATH_MTSD_LABEL)
    # Collect mapping from original MTSD labels to new class index
    mtsd_label_to_class_index: Dict[str, int] = {}

    # TODO(chawins@): For each dataset, we should have (1) a mapping from
    # original MTSD labels to new class index, (2) a mapping from new class
    # index to class name.

    for idx, row in label_map.iterrows():
        new_target = OLD_TO_NEW_LABELS.get(row["target"])
        if "orig" in dataset_name:
            # Use original MTSD labels ("orig" mode)
            mtsd_label_to_class_index[row["sign"]] = idx
        elif "100" in dataset_name:
            try:
                label_idx = class_names.index(row["sign"])
            except ValueError:
                label_idx = bg_idx
            mtsd_label_to_class_index[row["sign"]] = label_idx
        elif new_target in class_names:
            if "no_color" in dataset_name:
                # Use shape MTSD labels ("no_color" mode)
                cat_idx = class_names.index(new_target)
            else:
                # Use shape + color MTSD labels ("color" mode)
                cat_idx = TS_COLOR_OFFSET_DICT[new_target]
                color_list = TS_COLOR_DICT[new_target]
                if len(color_list) > 0:
                    cat_idx += color_list.index(row["color"])
            mtsd_label_to_class_index[row["sign"]] = cat_idx

    # Get all JSON files
    json_files: List[str] = [
        str(f)
        for f in label_path.iterdir()
        if f.is_file() and f.suffix == ".json"
    ]

    return {
        "duplicate_files_df": duplicate_files_df,
        "mtsd_label_to_class_index": mtsd_label_to_class_index,
        "json_files": json_files,
    }


def get_mtsd_dict(
    split: str,
    data_path: str,
    json_files: Optional[List[str]] = None,
    duplicate_files_df: Optional[pd.DataFrame] = None,
    mtsd_label_to_class_index: Optional[Dict[str, int]] = None,
    bg_class: int = 10,
    ignore_bg_class: bool = False,
    skip_bg_only: bool = True,
    **kwargs,
) -> List[DetectronSample]:
    """Get MTSD dataset as list of samples in Detectron2 format.

    Args:
        split: Dataset split to consider.
        data_path: Base path to dataset. Defaults to "~/data/".
        json_files: List of paths to JSON files each of which contains original
            annotation of one image.
        duplicate_files_df: DataFrame of duplicated files between MTSD and
            Mapillary Vistas.
        mtsd_label_to_class_index: Dictionary that maps original MTSD labels to
            new class index.
        bg_class_id: Background class index.
        ignore_bg_class: Whether to ignore background class (last class index).
            Defaults to False.

    Raises:
        ValueError: split is not among _ALLOWED_SPLITS.

    Returns:
        List of MTSD samples in Detectron2 format.
    """
    del kwargs  # Unused
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split must be among {_ALLOWED_SPLITS}, but it is {split}!"
        )

    if (
        json_files is None
        or duplicate_files_df is None
        or mtsd_label_to_class_index is None
    ):
        raise ValueError(
            "Some MTSD metadata are missing! Please use get_mtsd_anno() to get "
            "all the neccessary metadata."
        )

    dpath: pathlib.Path = pathlib.Path(data_path)
    filenames = _readlines(str(dpath / "splits" / (split + ".txt")))
    filenames = set(filenames)
    dataset_dicts: List[DetectronSample] = []
    duplicate_file_names = set(duplicate_files_df["filename"].values)

    for idx, json_file in enumerate(tqdm(json_files)):

        filename: str = json_file.split("/")[-1].split(".")[0]
        # Skip samples not in this split
        if filename not in filenames:
            continue
        jpg_filename: str = f"{filename}.jpg"
        # Skip samples that appear in Mapillary Vistas
        if jpg_filename in duplicate_file_names:
            continue

        # Read JSON files
        with open(json_file, "r", encoding="utf-8") as file:
            anno: Dict[str, Any] = json.load(file)

        height, width = anno["height"], anno["width"]
        record: DetectronSample = {
            "file_name": str(dpath / split / jpg_filename),
            "image_id": idx,
            "width": width,
            "height": height,
        }

        # Populate record or sample with its objects
        objs: List[Dict[str, Any]] = []
        non_bg_found = False
        for obj in anno["objects"]:
            class_index: int = mtsd_label_to_class_index.get(
                obj["label"], bg_class
            )
            # Remove labels for small or "other" objects
            if ignore_bg_class and class_index == bg_class:
                continue
            if class_index != bg_class:
                non_bg_found = True
            obj: Dict[str, Any] = {
                "bbox": [
                    obj["bbox"]["xmin"],
                    obj["bbox"]["ymin"],
                    obj["bbox"]["xmax"],
                    obj["bbox"]["ymax"],
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_index,
                "has_reap": False,
                "keypoints": [0] * _NUM_KEYPOINTS * 3,
                "ct_coeffs": None,
                "poly_coeffs": None,
            }
            objs.append(obj)

        # Skip images with no object of interest
        if (skip_bg_only and not non_bg_found) or len(objs) == 0:
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_mtsd(
    base_path: str = "~/data/",
    dataset_name: str = "mtsd",
) -> None:
    """Register MTSD dataset on Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "~/data/".
        dataset_name: Dataset name along with modifiers. Defaults to "mtsd".
    """
    data_path = os.path.join(base_path, "mtsd_v2_fully_annotated")
    (
        _,
        _,
        use_orig_labels,
        ignore_bg_class,
        skip_bg_only,
        _,
        split,
    ) = parse_dataset_name(dataset_name)
    if split is not None:
        dataset_name = "-".join(dataset_name.split("-")[:-1])
    class_names: list[str] = list(
        DATASET_METADATA[dataset_name]["class_name"].values()
    )
    mtsd_anno: dict[str, Any] = get_mtsd_anno(data_path, dataset_name)
    bg_class: int = len(class_names) - 1

    label_map: pd.DataFrame = mtsd_anno["mtsd_label_to_class_index"]
    if use_orig_labels:
        thing_classes = label_map["sign"].tolist()
    else:
        thing_classes = class_names
        if ignore_bg_class:
            thing_classes = thing_classes[:-1]

    metadata = {
        "thing_classes": thing_classes,
        "keypoint_names": [f"p{i}" for i in range(_NUM_KEYPOINTS)],
        "keypoint_flip_map": [
            (f"p{i}", f"p{i}") for i in range(_NUM_KEYPOINTS)
        ],
        "obj_dim_dict": DATASET_METADATA[dataset_name],
        "bg_class": bg_class,
    }
    # Register dataset without split to keep metadata
    DatasetCatalog.register(dataset_name, lambda x: [])
    MetadataCatalog.get(dataset_name).set(**metadata)

    for split in _ALLOWED_SPLITS:
        dataset_with_split: str = f"{dataset_name}_{split}"
        DatasetCatalog.register(
            dataset_with_split,
            lambda s=split: get_mtsd_dict(
                split=s,
                data_path=data_path,
                bg_class=bg_class,
                ignore_bg_class=ignore_bg_class,
                skip_bg_only=skip_bg_only,
                **mtsd_anno,
            ),
        )
        MetadataCatalog.get(dataset_with_split).set(**metadata)
