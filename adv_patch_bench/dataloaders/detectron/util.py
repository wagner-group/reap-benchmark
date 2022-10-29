"""Setup data loaders."""

import os
from typing import Any, Dict, List, Optional

import detectron2
import pandas as pd
from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.dataloaders.detectron import mapillary, mtsd, reap
from adv_patch_bench.utils.types import DetectronSample
from hparams import DATASETS

_LOAD_DATASET = {
    "reap": reap.get_reap_dict,
    "synthetic": reap.get_reap_dict,
    "mapillary": mapillary.get_mapillary_dict,
    "mtsd": mtsd.get_mtsd_dict,
}


def get_dataset(config_eval: Dict[str, Any]) -> List[DetectronSample]:
    """Load dataset in Detectron2 format (list of samples, i.e., dictionaries).

    This function may be unneccessary since we can globally call
    detectron2.data.DatasetCatalog to get data_dicts directly.

    Args:
        config_eval: Evaluation config.

    Raises:
        NotImplementedError: Invalid dataset.

    Returns:
        Dataset as list of dictionaries.
    """
    dataset: str = config_eval["dataset"]
    base_dataset: str = dataset.split("_")[0]
    split: str = config_eval["dataset_split"]
    base_path: str = os.path.expanduser(config_eval["data_dir"])
    # This assumes that dataset has been registered before
    class_names: List[str] = detectron2.data.MetadataCatalog.get(
        base_dataset
    ).get("thing_classes")
    bg_class_id: int = len(class_names) - 1
    # Load annotation if specified
    anno_df: Optional[pd.DataFrame] = None
    if config_eval["annotated_signs_only"]:
        anno_df = reap_util.load_annotation_df(config_eval["tgt_csv_filepath"])

    if base_dataset not in _LOAD_DATASET:
        raise NotImplementedError(
            f"Dataset {base_dataset} is not implemented! Only {DATASETS} are "
            "available."
        )

    # Load additional metadata for MTSD
    mtsd_anno: Dict[str, Any] = mtsd.get_mtsd_anno(
        base_path, config_eval["use_color"], "orig" in dataset, class_names
    )

    data_dict: List[DetectronSample] = _LOAD_DATASET[base_dataset](
        split=split,
        base_path=base_path,
        bg_class_id=bg_class_id,
        ignore_bg_class=False,
        anno_df=anno_df,
        **mtsd_anno,
    )
    return data_dict


def register_dataset(config_eval: Dict[str, Any]) -> None:
    """Register dataset for Detectron2.

    TODO(yolo): Combine with YOLO dataloader.

    Args:
        config_eval: Dictionary of eval config.
    """
    dataset: str = config_eval["dataset"]
    base_dataset: str = dataset.split("_")[0]
    # Get data path
    base_data_path: str = os.path.expanduser(config_eval["data_dir"])
    use_color: bool = config_eval["use_color"]

    # Load annotation if specified
    anno_df: Optional[pd.DataFrame] = None
    if config_eval["annotated_signs_only"]:
        anno_df = reap_util.load_annotation_df(config_eval["tgt_csv_filepath"])

    if base_dataset in ("reap", "synthetic"):
        # Our synthetic benchmark is also based on samples in REAP
        reap.register_reap(
            base_path=base_data_path,
            synthetic=base_dataset == "synthetic",
            anno_df=anno_df,
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
        )
