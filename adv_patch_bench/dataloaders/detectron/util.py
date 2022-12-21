"""Setup data loaders."""

from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Any, Dict, List, Optional

import detectron2
import pandas as pd
import torch
from detectron2.config import global_cfg
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO

from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.dataloaders.detectron import (
    custom_build,
    mapillary,
    mtsd,
    reap,
    reap_dataset_mapper,
)
from adv_patch_bench.utils.types import DetectronSample
from hparams import DATASETS

_LOAD_DATASET = {
    "reap": reap.get_reap_dict,
    "synthetic": reap.get_reap_dict,
    "mapillary": mapillary.get_mapillary_dict,
    "mtsd": mtsd.get_mtsd_dict,
}

log = logging.getLogger(__name__)


def _get_img_ids(dataset: str, obj_class: int) -> list[int]:
    """Get ids of images that contain desired object class."""
    metadata = detectron2.data.MetadataCatalog.get(dataset)
    if not hasattr(metadata, "json_file"):
        cache_path = os.path.join(
            global_cfg.OUTPUT_DIR, f"{dataset}_coco_format.json"
        )
        metadata.json_file = cache_path
        convert_to_coco_json(dataset, cache_path)

    json_file = PathManager.get_local_path(metadata.json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    img_ids: list[int] = coco_api.getImgIds(catIds=[obj_class])
    return img_ids


def _get_filename_from_id(
    data_dicts: list[DetectronSample], img_ids: list[int]
) -> list[str]:
    filenames: list[str] = []
    img_ids_set = set(img_ids)
    for data in data_dicts:
        if data["image_id"] in img_ids_set:
            filenames.append(data["file_name"].split("/")[-1])
    return filenames


def get_dataloader(
    config_base: dict[str, Any]
) -> tuple[torch.data.utils.DataLoader, set[str]]:
    """Get eval dataloader from base config."""
    dataset: str = config_base["dataset"]
    split_file_path: str = config_base["split_file_path"]

    # First, get list of file names to evaluate on
    data_dicts: List[DetectronSample] = detectron2.data.DatasetCatalog.get(
        config_base["dataset"]
    )
    split_file_names: set[str] = set()
    if split_file_path is not None:
        log.info("Loading file names from %s...", split_file_path)
        with open(split_file_path, "r", encoding="utf-8") as file:
            split_file_names = set(file.read().splitlines())

    # Filter only images with desired class when evaluating on REAP
    if dataset == "reap":
        img_ids = _get_img_ids(dataset, config_base["obj_class"])
        class_file_names = set(_get_filename_from_id(data_dicts, img_ids))
        split_file_names = split_file_names.intersection(class_file_names)

    dataloader = custom_build.build_detection_test_loader(
        data_dicts,
        mapper=reap_dataset_mapper.ReapDatasetMapper(
            global_cfg, is_train=False, img_size=config_base["img_size"]
        ),
        batch_size=config_base["batch_size"],
        num_workers=global_cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        split_file_names=split_file_names,
    )

    return dataloader, split_file_names


def get_dataset(config_base: Dict[str, Any]) -> List[DetectronSample]:
    """Load dataset in Detectron2 format (list of samples, i.e., dictionaries).

    This function may be unneccessary since we can globally call
    detectron2.data.DatasetCatalog to get data_dicts directly.

    Args:
        config_base: Evaluation config.

    Raises:
        NotImplementedError: Invalid dataset.

    Returns:
        Dataset as list of dictionaries.
    """
    dataset: str = config_base["dataset"]
    base_dataset: str = dataset.split("_")[0]
    split: str = config_base["dataset_split"]
    base_path: str = os.path.expanduser(config_base["data_dir"])
    # This assumes that dataset has been registered before
    class_names: List[str] = detectron2.data.MetadataCatalog.get(
        base_dataset
    ).get("thing_classes")
    bg_class: int = len(class_names) - 1
    # Load annotation if specified
    anno_df: Optional[pd.DataFrame] = None
    if config_base["annotated_signs_only"]:
        anno_df = reap_util.load_annotation_df(config_base["tgt_csv_filepath"])

    if base_dataset not in _LOAD_DATASET:
        raise NotImplementedError(
            f"Dataset {base_dataset} is not implemented! Only {DATASETS} are "
            "available."
        )

    # Load additional metadata for MTSD
    mtsd_anno: Dict[str, Any] = mtsd.get_mtsd_anno(
        base_path, config_base["use_color"], "orig" in dataset, class_names
    )

    data_dict: List[DetectronSample] = _LOAD_DATASET[base_dataset](
        split=split,
        data_path=base_path,
        bg_class=bg_class,
        ignore_bg_class=False,
        anno_df=anno_df,
        img_size=config_base["img_size"],
        **mtsd_anno,
    )
    return data_dict


def register_dataset(config_base: Dict[str, Any]) -> None:
    """Register dataset for Detectron2.

    TODO(yolo): Combine with YOLO dataloader.

    Args:
        config_base: Dictionary of eval config.
    """
    dataset: str = config_base["dataset"]
    base_dataset: str = dataset.split("_")[0]
    # Get data path
    base_data_path: str = os.path.expanduser(config_base["data_dir"])
    use_color: bool = config_base["use_color"]

    # Load annotation if specified
    anno_df: Optional[pd.DataFrame] = None
    if config_base.get("annotated_signs_only", False):
        anno_df = reap_util.load_annotation_df(config_base["tgt_csv_filepath"])

    log.info("Registering %s dataset...", base_dataset)
    if base_dataset in ("reap", "synthetic"):
        # Our synthetic benchmark is also based on samples in REAP
        reap.register_reap(
            base_path=base_data_path,
            synthetic=base_dataset == "synthetic",
            anno_df=anno_df,
            img_size=config_base["img_size"],
        )
    elif base_dataset == "mtsd":
        mtsd.register_mtsd(
            base_path=base_data_path,
            use_color=use_color,
            use_mtsd_original_labels="orig" in dataset,
            ignore_bg_class=False,
        )
    elif base_dataset == "mapillary":
        mapillary.register_mapillary(
            base_path=base_data_path,
            use_color=use_color,
            ignore_bg_class=False,
            anno_df=anno_df,
            img_size=config_base["img_size"],
        )
    else:
        raise NotImplementedError(f"Dataset {base_dataset} is not supported!")
