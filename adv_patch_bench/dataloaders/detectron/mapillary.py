"""Register and load Mapillary Vistas dataset."""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.utils.types import DetectronSample, SizePx
from hparams import DATASET_METADATA, LABEL_LIST

_ALLOWED_SPLITS = ("train", "test", "combined")
_NUM_KEYPOINTS = 4

logger = logging.getLogger(__name__)


def get_mapillary_dict(
    split: str = "train",
    data_path: str = "./",
    bg_class: int = 10,
    ignore_bg_class: bool = False,
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
    **kwargs,
) -> list[DetectronSample]:
    """Get Mapillary Vistas dataset as list of samples in Detectron2 format.

    Args:
        split: Dataset split to consider.
        base_path: Base path to dataset.
        bg_class_id: Background class index.
        ignore_bg_class: Whether to exclude background objects from labels.
            Defaults to False. ignore_bg_class should not be True when running
            evaluation because this means background objects will not have a
            ground-truth bounding box. This will wrongly incur high FPR.
        anno_df: Transform annotation. Defaults to None.
        img_size: Desired image size (height, width). Note that images are not
            resized here but by DatasetMapper instead. So if a corresponding
            DatasetMapper is not called properly, bbox and keypoints may be
            wrong. Defaults to None.

    Raises:
        ValueError: split is not among _ALLOWED_SPLITS.

    Returns:
        List of Mapillary Vistas samples in Detectron2 format.
    """
    _ = kwargs  # Unused
    logger.info("Registering %s Mapillary Vistas data at %s", split, data_path)
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split must be among {_ALLOWED_SPLITS}, but it is {split}!"
        )
    if img_size is not None:
        new_height, new_width = img_size

    mapillary_split: Dict[str, str] = {
        "train": "training",
        "test": "validation",
        "combined": "combined",
    }[split]
    bpath: pathlib.Path = pathlib.Path(data_path)
    label_path: pathlib.Path = bpath / mapillary_split / "detectron_labels"
    img_path: pathlib.Path = bpath / mapillary_split / "images"

    dataset_dicts = []
    label_files: List[str] = [
        str(f) for f in label_path.iterdir() if f.is_file()
    ]
    label_files = sorted(label_files)
    img_df: Optional[pd.DataFrame] = None

    for idx, label_file in enumerate(tqdm(label_files)):

        filename: str = label_file.split(".txt")[0].split("/")[-1]
        jpg_filename: str = f"{filename}.jpg"
        if anno_df is not None:
            img_df = anno_df[anno_df["filename"] == jpg_filename]

        with open(label_file, "r", encoding="utf-8") as file:
            labels: List[str] = file.readlines()
            labels = [a.strip() for a in labels]

        width: float = float(labels[0].split(",")[5])
        height: float = float(labels[0].split(",")[6])
        record: DetectronSample = {
            "file_name": str(img_path / jpg_filename),
            "image_id": idx,
            "width": new_width if img_size is not None else width,
            "height": new_height if img_size is not None else height,
        }

        scales, padding = [1, 1], [0] * 4
        if img_size is not None:
            _, scales, padding = img_util.resize_and_pad(
                orig_size=(height, width),
                resize_size=img_size,
                pad_size=img_size,
                keep_aspect_ratio=True,
                return_params=True,
            )

        # Populate record or sample with its objects
        objs: List[Dict[str, Any]] = []
        for orig_obj in labels:
            class_id, xmin, ymin, xmax, ymax, _, _, obj_id = orig_obj.split(",")
            xmin, ymin, xmax, ymax = [
                float(x) for x in [xmin, ymin, xmax, ymax]
            ]
            # We choose to transform (shift and scale) bbox here instead of
            # automatically by detectron2 evaluator because it does not support
            # padding. Another option is to do it in DatasetMapper, but it does
            # not get called by COCO evaluator.
            ymin = ymin * scales[0] + padding[1]
            ymax = ymax * scales[0] + padding[1]
            xmin = xmin * scales[1] + padding[0]
            xmax = xmax * scales[1] + padding[0]
            class_id, obj_id = int(class_id), int(obj_id)

            obj_df = None
            if img_df is not None:
                obj_df = img_df[img_df["object_id"] == obj_id]
                if obj_df.empty:
                    # If we want results on annotated signs, we set the class of
                    # the unannotated ones to "other" or background class.
                    # We cannot simply remove them because it will incur a false
                    # positive.
                    class_id = bg_class

            # Remove "other" objects
            if ignore_bg_class and class_id == bg_class:
                continue

            assert 0 <= class_id <= bg_class, (
                f"class_id {class_id} seems to be out of range ({bg_class} "
                "max) Something went wrong."
            )

            # Annotation for each object
            obj: Dict[str, Any] = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_id,
                "object_id": obj_id,
                "has_reap": False,
                "keypoints": [0] * _NUM_KEYPOINTS * 3,
                "relight_coeffs": None,
            }

            if obj_df is not None and not obj_df.empty and class_id != bg_class:
                # Include REAP annotation if exists
                tgt_points = obj_df["tgt_points"].values[0]
                keypoints = []
                if len(tgt_points) > _NUM_KEYPOINTS:
                    print(
                        f"Each object should have at most {_NUM_KEYPOINTS} "
                        f"keypoints but found {len(tgt_points)}!"
                    )
                    print(obj_df)
                    assert class_id == bg_class
                    tgt_points = tgt_points[:_NUM_KEYPOINTS]
                for tgt in tgt_points:
                    keypoints.extend(
                        (
                            tgt[0] * scales[1] + padding[0],
                            tgt[1] * scales[0] + padding[1],
                            2,
                        )
                    )
                if len(tgt_points) == _NUM_KEYPOINTS - 1:
                    keypoints.extend(
                        (
                            sum(keypoints[i] for i in [0, 3, 6]) / 3,
                            sum(keypoints[i] for i in [1, 4, 7]) / 3,
                            2,
                        )
                    )
                assert len(keypoints) == _NUM_KEYPOINTS * 3 and all(
                    keypoints[i] == 2 for i in [2, 5, 8, 11]
                )
                obj["keypoints"] = keypoints
                if "ct_coeffs" in obj_df.columns:
                    obj["ct_coeffs"] = obj_df["ct_coeffs"].values
                if "poly_coeffs" in obj_df.columns:
                    obj["poly_coeffs"] = obj_df["poly_coeffs"].values
                obj["has_reap"] = True

            objs.append(obj)

        # Skip images with no object of interest
        if len(objs) == 0 or all(o["category_id"] == bg_class for o in objs):
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_mapillary(
    base_path: str = "~/data/",
    use_color: bool = False,
    ignore_bg_class: bool = False,
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
) -> None:
    """Register Mapillary Vistas dataset on Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "~/data/".
        use_color: Whether color is used as part of labels. Defaults to False.
        ignore_bg_class: Whether to ignore background class (last class index).
            Defaults to False.
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
        img_size: Desired image size (height, width). Note that images are not
            resized here but by DatasetMapper instead. So if a corresponding
            DatasetMapper is not called properly, bbox and keypoints may be
            wrong. Defaults to None.
    """
    color: str = "color" if use_color else "no_color"
    dataset: str = f"mapillary_{color}"
    data_path = os.path.join(base_path, "mapillary_vistas", color)

    class_names: List[str] = LABEL_LIST[dataset]
    bg_class: int = len(class_names) - 1
    thing_classes: List[str] = class_names
    if ignore_bg_class:
        thing_classes = thing_classes[:-1]

    for split in _ALLOWED_SPLITS:
        dataset_with_split: str = f"mapillary_{color}_{split}"
        DatasetCatalog.register(
            dataset_with_split,
            lambda s=split: get_mapillary_dict(
                split=s,
                data_path=data_path,
                bg_class=bg_class,
                ignore_bg_class=ignore_bg_class,
                anno_df=anno_df,
                img_size=img_size,
            ),
        )
        MetadataCatalog.get(dataset_with_split).set(
            thing_classes=thing_classes,
            keypoint_names=[f"p{i}" for i in range(_NUM_KEYPOINTS)],
            keypoint_flip_map=[
                (f"p{i}", f"p{i}") for i in range(_NUM_KEYPOINTS)
            ],
            obj_dim_dict=DATASET_METADATA[f"mapillary_{color}"],
            bg_class=bg_class,
        )
