"""Register and load REAP benchmark as well as its synthetic version."""

from __future__ import annotations

import os

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog

from adv_patch_bench.dataloaders.detectron import mapillary
from adv_patch_bench.utils.types import DetectronSample, SizePx
from hparams import LABEL_LIST, DATASET_METADATA

_NUM_KEYPOINTS = 4


def get_reap_dict(
    data_path: str = "~/data/",
    bg_class: int = 10,
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
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
        ignore_bg_class=False,
        anno_df=anno_df,
        img_size=img_size,
    )
    return data_dict


def register_reap(
    base_path: str = "~/data/",
    synthetic: bool = False,
    anno_df: pd.DataFrame | None = None,
    img_size: SizePx | None = None,
) -> None:
    """Register REAP dataset in Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "./data/".
        synthetic: Whether to use synthetic version. Defaults to False.
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
        img_size: Desired image size (height, width). Note that images are not
            resized here but by DatasetMapper instead. So if a corresponding
            DatasetMapper is not called properly, bbox and keypoints may be
            wrong. Defaults to None.
    """
    dataset_name: str = "synthetic" if synthetic else "reap"
    class_names: list[str] = LABEL_LIST[dataset_name]
    # Get index of background or "other" class
    bg_class: int = len(class_names) - 1
    base_path = os.path.expanduser(base_path)
    data_path: str = os.path.join(base_path, "mapillary_vistas", "no_color")

    DatasetCatalog.register(
        dataset_name,
        lambda: get_reap_dict(
            data_path=data_path,
            bg_class=bg_class,
            anno_df=anno_df,
            img_size=img_size,
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        thing_classes=class_names,
        keypoint_names=[f"p{i}" for i in range(_NUM_KEYPOINTS)],
        keypoint_flip_map=[(f"p{i}", f"p{i}") for i in range(_NUM_KEYPOINTS)],
        obj_dim_dict=DATASET_METADATA["reap"],
        bg_class=bg_class,
    )
