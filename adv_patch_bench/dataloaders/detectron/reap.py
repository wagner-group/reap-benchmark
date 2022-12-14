"""Register and load REAP benchmark as well as its synthetic version."""

import os
from typing import List, Optional

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog

from adv_patch_bench.dataloaders.detectron import mapillary
from adv_patch_bench.utils.types import DetectronSample
from hparams import LABEL_LIST


def get_reap_dict(
    base_path: str,
    bg_class_id: int,
    anno_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> List[DetectronSample]:
    """Load REAP dataset through Mapillary Vistas loader.

    See mapillary.get_mapillary_dict() for args and returns.
    """
    del kwargs  # Unused
    data_dict = mapillary.get_mapillary_dict(
        split="combined",
        base_path=base_path,
        bg_class_id=bg_class_id,
        ignore_bg_class=False,
        anno_df=anno_df,
    )
    return data_dict


def register_reap(
    base_path: str = "~/data/",
    synthetic: bool = False,
    anno_df: Optional[pd.DataFrame] = None,
) -> None:
    """Register REAP dataset in Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "./data/".
        synthetic: Whether to use synthetic version. Defaults to False.
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
    """
    dataset_name: str = "synthetic" if synthetic else "reap"
    class_names: List[str] = LABEL_LIST[dataset_name]
    # Get index of background or "other" class
    bg_class_id: int = len(class_names) - 1
    base_path = os.path.expanduser(base_path)
    data_path: str = os.path.join(base_path, "mapillary_vistas", "no_color")

    DatasetCatalog.register(
        dataset_name,
        lambda: get_reap_dict(
            data_path,
            bg_class_id,
            anno_df,
        ),
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=class_names)
