"""Get default global config."""

from __future__ import annotations

import argparse
import os
from typing import Any

import detectron2
from detectron2.config import CfgNode, LazyConfig
from detectron2.engine import default_setup
from detectron2.utils import comm
from omegaconf import OmegaConf

from hparams import Metadata


def _cfgnode_to_dict(cfg_node: CfgNode, cfg_dict: dict[str, Any] | None = None):
    if cfg_dict is None:
        cfg_dict = {}
    for k, v in cfg_node.items():
        if isinstance(v, CfgNode):
            cfg_dict[k] = {}
            _cfgnode_to_dict(v, cfg_dict=cfg_dict[k])
        else:
            cfg_dict[k] = v
    return cfg_dict


def get_default_cfg(config_base: dict[str, Any]) -> CfgNode:
    """Get default config.

    Args:
        config_base: Config base.
    """
    # Setup detectron default config
    if "detrex" in config_base["model_name"]:
        d2_cfg = detectron2.config.get_cfg()
        # detrex uses python config instead of yaml
        cfg = LazyConfig.load(config_base["config_file"])
        cfg = LazyConfig.apply_overrides(cfg, config_base["opts"])
        d2_cfg = _cfgnode_to_dict(d2_cfg)
        cfg = OmegaConf.merge(d2_cfg, cfg)
        cfg.MODEL.META_ARCHITECTURE = "detrex"
        # detrex sets input format differently (default is "RGB")
        cfg.INPUT.FORMAT = cfg.model.input_format
    else:
        if "yolof" in config_base["model_name"]:
            # TODO(enhancement): Combine get_cfg with a wrapper.
            from yolof.config import get_cfg

            cfg = get_cfg()
        elif "yolov7" in config_base["model_name"]:
            # We import here to avoid backbone being registered twice
            from yolov7.config import add_yolo_config

            cfg = detectron2.config.get_cfg()
            add_yolo_config(cfg)
        else:
            cfg = detectron2.config.get_cfg()
        cfg.merge_from_file(config_base["config_file"])
        cfg.merge_from_list(config_base["opts"])

    return cfg


def setup_detectron_cfg(
    config: dict[str, dict[str, Any]],
    is_train: bool = False,
) -> detectron2.config.CfgNode:
    """Create configs and perform basic setups."""
    config_base = config["base"]
    dataset: str = config_base["dataset"]
    split: str = config_base["dataset_split"]
    num_classes = len(Metadata.get(dataset).class_names)

    # Set default path to load adversarial patch
    if not config_base["adv_patch_path"]:
        config_base["adv_patch_path"] = os.path.join(
            config_base["save_dir"], "adv_patch.pkl"
        )

    cfg = get_default_cfg(config_base)

    # Copy dataset from args
    cfg.DATASETS.TEST = (f"{dataset}_{'val' if is_train else split}",)
    cfg.DATASETS.TRAIN = (config_base["train_dataset"],)
    cfg.SOLVER.IMS_PER_BATCH = config_base["batch_size"]
    if not is_train:
        # Turn off augmentation for testing. Otherwise, leave it to config file.
        cfg.INPUT.CROP.ENABLED = False
    cfg.DATALOADER.NUM_WORKERS = config_base["workers"]
    cfg.eval_mode = config_base["eval_mode"]
    cfg.obj_class = config_base["obj_class"]
    # Assume that "other" class is always last
    cfg.other_catId = num_classes - 1
    config_base["other_sign_class"] = cfg.other_catId
    # cfg.conf_thres = config_base["conf_thres"]

    # Set detectron image size from argument
    # Let img_size be (1536, 2048). This tries to set the smaller side of the
    # image to 2048, but the longer side cannot exceed 2048. The result of this
    # is that every image has its longer side (could be width or height) 2048.
    cfg.INPUT.MAX_SIZE_TEST = max(config_base["img_size"])
    cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MAX_SIZE_TEST

    # Model config
    cfg.OUTPUT_DIR = config_base["save_dir"]
    cfg.SEED = config_base["seed"]
    weight_path = config_base["weights"]
    if isinstance(config_base["weights"], list):
        weight_path = weight_path[0]
    cfg.MODEL.WEIGHTS = weight_path

    # Replace SynBN with BN when running on one GPU
    _find_and_set_bn(cfg)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if "YOLOF" in cfg.MODEL:
        cfg.MODEL.YOLOF.DECODER.NUM_CLASSES = num_classes
    elif "YOLO" in cfg.MODEL:
        cfg.MODEL.YOLO.CLASSES = num_classes
    elif "detrex" in config_base["model_name"]:
        cfg.model.num_classes = num_classes
        cfg.model.criterion.num_classes = num_classes

    if isinstance(cfg, CfgNode):
        cfg.freeze()
    # Set cfg as global variable so we can avoid passing cfg around
    detectron2.config.set_global_cfg(cfg)
    config_base.pop("config_file")  # Remove to avoid logging
    if "yolov7" in config_base["model_name"]:
        from yolov7.utils.d2overrides import default_setup as y7_default_setup

        y7_default_setup(cfg, argparse.Namespace(**config_base))
    else:
        default_setup(cfg, argparse.Namespace(**config_base))

    return cfg


def _find_and_set_bn(cfg_node):
    """Replace SyncBN with BN in config if not in distributed mode."""
    if comm.get_world_size() > 1:
        return
    if not isinstance(cfg_node, detectron2.config.CfgNode):
        return
    for key, value in cfg_node.items():
        if isinstance(value, str):
            cfg_node[key] = value.replace("SyncBN", "BN")
        _find_and_set_bn(value)
