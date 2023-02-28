"""Register and load REAP benchmark as well as its synthetic version."""

from __future__ import annotations

import logging
import os

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog

from adv_patch_bench.dataloaders.detectron import mapillary
from adv_patch_bench.utils.argparse import parse_dataset_name
from adv_patch_bench.utils.types import DetectronSample, SizePx
from hparams import DATASET_METADATA

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


def register_reap(
    base_path: str = "~/data/",
    dataset_name: str = "reap",
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
) -> None:
    """Register REAP dataset in Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "./data/".
        dataset_name: Full dataset name. Defaults to "reap".
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
        img_size: Desired image size (height, width). Note that images are not
            resized here but by DatasetMapper instead. So if a corresponding
            DatasetMapper is not called properly, bbox and keypoints may be
            wrong. Defaults to None.
    """
    class_names: list[str] = list(
        DATASET_METADATA[dataset_name]["class_name"].values()
    )
    # Get index of background or "other" class
    bg_class: int = len(class_names) - 1
    base_path = os.path.expanduser(base_path)
    base_dataset, use_color, _, nobg, _, num_classes, _ = parse_dataset_name(
        dataset_name
    )
    assert base_dataset == "reap", (
        f"Dataset name must start with 'reap' but is {base_dataset}!"
    )
    if num_classes is not None:
        modifier = str(num_classes)
    else:
        modifier = "color" if use_color else "no_color"
    data_path: str = os.path.join(base_path, "mapillary_vistas", modifier)
    logger.info("Registering REAP dataset at %s", data_path)

    DatasetCatalog.register(
        dataset_name,
        lambda: get_reap_dict(
            data_path=data_path,
            bg_class=bg_class,
            anno_df=anno_df,
            img_size=img_size,
            ignore_bg_class=nobg,
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=class_names,
        keypoint_names=[f"p{i}" for i in range(_NUM_KEYPOINTS)],
        keypoint_flip_map=[(f"p{i}", f"p{i}") for i in range(_NUM_KEYPOINTS)],
        obj_dim_dict=DATASET_METADATA[dataset_name],
        bg_class=bg_class,
    )
