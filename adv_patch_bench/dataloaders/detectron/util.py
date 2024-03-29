"""Setup data loaders."""

from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from detectron2.config import global_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
from torch.utils.data.sampler import Sampler

from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.dataloaders.detectron import (
    custom_build,
    custom_sampler,
    mapillary,
    mtsd,
    mtsd_dataset_mapper,
    realism,
    reap,
    reap_dataset_mapper,
)
from adv_patch_bench.utils.types import DetectronSample
from hparams import Metadata

logger = logging.getLogger(__name__)


def _get_img_ids(dataset: str, obj_class: int | None) -> list[int]:
    """Get ids of images that contain desired object class."""
    metadata = MetadataCatalog.get(dataset)
    if not hasattr(metadata, "json_file"):
        logger.info(
            "COCO json file not found in MetadataCatalog for %s dataset. "
            "Converting dataset to COCO format...",
            dataset,
        )
        cache_path = os.path.join(
            global_cfg.OUTPUT_DIR, f"{dataset}_coco_format.json"
        )
        metadata.set(json_file=cache_path)
        metadata.json_file = cache_path
        convert_to_coco_json(dataset, cache_path)

    json_file = PathManager.get_local_path(metadata.json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    cat_ids = []
    if obj_class is not None and obj_class >= 0:
        cat_ids = [obj_class]
    img_ids: list[int] = coco_api.getImgIds(catIds=cat_ids)
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
    config_base: dict[str, Any],
    sampler: Sampler | str | None = None,
) -> tuple[torch.data.utils.DataLoader, set[str]]:
    """Get eval dataloader from base config."""
    dataset: str = config_base["dataset"]
    base_dataset: str = Metadata.parse_dataset_name(dataset).name
    split_file_path: str = config_base["split_file_path"]
    metadata = MetadataCatalog.get(dataset)

    # First, get list of file names to evaluate on
    if not hasattr(metadata, "data_dict"):
        data_dicts: list[DetectronSample] = DatasetCatalog.get(dataset)
        metadata.set(data_dict=data_dicts)
    else:
        data_dicts = metadata.data_dict
    split_file_names: set[str] = set()
    if split_file_path is not None:
        logger.info("Loading file names from %s...", split_file_path)
        with open(split_file_path, "r", encoding="utf-8") as file:
            split_file_names = set(file.read().splitlines())

    # Filter only images with desired class when evaluating on REAP
    if "reap" in base_dataset:
        img_ids = _get_img_ids(dataset, config_base["obj_class"])
        class_file_names = set(_get_filename_from_id(data_dicts, img_ids))
        split_file_names = split_file_names.intersection(class_file_names)

    if sampler == "shuffle":
        logger.info("Using shuffle sampler...")
        num_samples: int = (
            len(split_file_names)
            if split_file_path is not None
            else len(data_dicts)
        )
        sampler = custom_sampler.ShuffleInferenceSampler(num_samples)

    if any(d in base_dataset for d in ("reap", "synthetic")):
        mapper = reap_dataset_mapper.ReapDatasetMapper(
            global_cfg, is_train=False, img_size=config_base["img_size"]
        )
    else:
        mapper = mtsd_dataset_mapper.MtsdDatasetMapper(
            global_cfg,
            config_base,
            is_train=False,
            img_size=config_base["img_size"],
        )

    dataloader = custom_build.build_detection_test_loader(
        data_dicts,
        mapper=mapper,
        batch_size=config_base["batch_size"],
        num_workers=global_cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        split_file_names=split_file_names,
        sampler=sampler,
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
    return DatasetCatalog.get(config_base["dataset"])


def register_dataset(config_base: Dict[str, Any]) -> None:
    """Register dataset for Detectron2.

    Args:
        config_base: Dictionary of eval config.
    """
    dataset: str = config_base["dataset"]
    base_dataset: str = Metadata.parse_dataset_name(dataset).name

    # Load annotation if specified
    anno_df: Optional[pd.DataFrame] = None
    if config_base.get("annotated_signs_only", False):
        anno_df = reap_util.load_annotation_df(
            Metadata.get(dataset).annotation_path
        )

    logger.info("Registering %s dataset...", base_dataset)
    if any(name in base_dataset for name in ("reap", "synthetic")):
        # Our synthetic benchmark is also based on samples in REAP
        reap.register_reap_syn(
            dataset_name=dataset,
            anno_df=anno_df,
            img_size=config_base["img_size"],
        )
    elif base_dataset == "mtsd":
        mtsd.register_mtsd(dataset_name=dataset)
    elif base_dataset == "mapillary":
        mapillary.register_mapillary(
            dataset_name=dataset,
            anno_df=anno_df,
            img_size=config_base["img_size"],
        )
    elif base_dataset == "realism":
        realism.register_realism(dataset_name=dataset)
    else:
        raise NotImplementedError(f"Dataset {base_dataset} is not supported!")
