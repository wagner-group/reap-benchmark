"""Register and load REAP benchmark as well as its synthetic version."""

from __future__ import annotations

import logging

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog

from adv_patch_bench.dataloaders.detectron import mapillary
from adv_patch_bench.utils.types import DetectronSample, SizePx
from hparams import Metadata

_NUM_KEYPOINTS = 4

logger = logging.getLogger(__name__)


def get_reap_dict(
    data_path: str = "~/data/",
    bg_class: int = 10,
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
    ignore_bg_class: bool = False,
    **kwargs,
) -> list[DetectronSample]:
    """Load REAP dataset through Mapillary Vistas loader.

    See mapillary.get_mapillary_dict() for args and returns.
    """
    _ = kwargs  # Unused
    data_dict = mapillary.get_mapillary_dict(
        split="combined",
        data_path=data_path,
        bg_class=bg_class,
        ignore_bg_class=ignore_bg_class,
        anno_df=anno_df,
        img_size=img_size,
    )
    return data_dict


def register_reap_syn(
    dataset_name: str = "reap",
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
) -> None:
    """Register REAP dataset in Detectron2.

    Args:
        dataset_name: Full dataset name. Defaults to "reap".
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
        img_size: Desired image size (height, width). Note that images are not
            resized here but by DatasetMapper instead. So if a corresponding
            DatasetMapper is not called properly, bbox and keypoints may be
            wrong. Defaults to None.
    """
    metadata = Metadata.get(dataset_name)
    dataset_id = Metadata.parse_dataset_name(dataset_name)
    class_names: list[str] = list(metadata.class_names.values())
    # Get index of background or "other" class
    bg_class: int = len(class_names) - 1
    data_path = metadata.data_path
    logger.info("Registering REAP dataset at %s", data_path)

    DatasetCatalog.register(
        dataset_name,
        lambda: get_reap_dict(
            data_path=data_path,
            bg_class=bg_class,
            anno_df=anno_df,
            img_size=img_size,
            ignore_bg_class=dataset_id.ignore_bg_class,
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=class_names,
        keypoint_names=[f"p{i}" for i in range(_NUM_KEYPOINTS)],
        keypoint_flip_map=[(f"p{i}", f"p{i}") for i in range(_NUM_KEYPOINTS)],
        bg_class=bg_class,
    )
