"""Define argument and config parsing."""

from __future__ import annotations

import argparse
import ast
import logging
import os
import pathlib
import pickle
from typing import Any, Dict, List, Optional

import detectron2
import numpy as np
import yaml
from detectron2.config import CfgNode, LazyConfig
from detectron2.engine import default_argument_parser, default_setup
from detectron2.utils import comm
from omegaconf import OmegaConf

from hparams import DATASET_METADATA, DEFAULT_SYN_OBJ_DIR, INTERPS

_TRANSFORM_PARAMS: List[str] = [
    "interp",
    "reap_geo_method",
    "reap_relight_method",
    "reap_relight_percentile",
    "reap_relight_polynomial_degree",
    "syn_obj_width_px",
    "syn_rotate",
    "syn_scale",
    "syn_translate",
    "syn_colorjitter",
    "syn_3d_dist",
]
logger = logging.getLogger(__name__)


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


def reap_args_parser(
    is_detectron: bool,
    root: Optional[str] = None,
    is_gen_patch: bool = False,
    is_train: bool = False,
) -> Dict[str, Any]:
    """Setup argparse for evaluation.

    TODO(enhancement): Improve argument documentation.

    Args:
        is_detectron: Whether we are evaluating dectectron model.
        root: Path to root or base directory. Defaults to current dir (None).

    Returns:
        Config dict containing two dicts, one for eval and the other for attack.
    """
    if root is None:
        root = os.getcwd()
    root = pathlib.Path(root)

    if is_detectron:
        parser = default_argument_parser()
    else:
        parser = argparse.ArgumentParser()

    # NOTE: we avoid flag name "--config-file" since it overlaps with detectron2
    parser.add_argument(
        "-e",
        "--exp-config-file",
        type=str,
        help="Path to YAML config file; overwrites args.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="./",
        help="Base output dir.",
    )
    parser.add_argument("--single-image", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Specify dataset name for evaluation or training.",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        required=False,
        default=None,
        help=(
            "Optionally specify dataset name for training if different from "
            "dataset."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (defaults to 1)."
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="drop",
        help=(
            "Set evaluation mode in COCO evaluator. Options: 'mtsd', 'drop' "
            "(default)."
        ),
    )
    parser.add_argument(
        "--interp",
        type=str,
        default="bilinear",
        help=(
            'Resample interpolation method: "nearest", "bilinear" (default), '
            '"bicubic".'
        ),
    )
    parser.add_argument(
        "--name", type=str, default=None, help="save to project/name"
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity level (0=ERROR, 1=INFO, 2=DEBUG).",
    )
    parser.add_argument("--seed", type=int, default=0, help="set random seed")
    parser.add_argument(
        "--padded-imgsz",
        type=str,
        default="3000,4000",
        help=(
            "Final image size including padding (height, width); "
            "Default: 3000,4000",
        ),
    )
    parser.add_argument(
        "--annotated-signs-only",
        action="store_true",
        help=(
            "If True, only calculate metrics on annotated signs. This is set to"
            " True by default when dataset is 'reap' or 'synthetic'."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="faster_rcnn",
        help="Set model name (faster_rcnn, yolov5, yolor).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=root / "yolov5s.pt",
        help="Path to PyTorch model weights.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=None,
        help=(
            "Set confidence threshold for detection."
            "Otherwise, threshold will be set to max f1 score."
        ),
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=None,
        help="Max number of images to eval on (default is entire dataset).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (per RANK in DDP mode).",
    )
    parser.add_argument(
        "--use-per-class-conf-thres",
        action="store_true",
        help="If True, expect/compute confidence score threshold per class.",
    )
    parser.add_argument(
        "--compute-conf-thres",
        action="store_true",
        help="If True, compute and save conf_thres in output_dir.",
    )

    # ====================== Specific to synthetic signs ===================== #
    parser.add_argument(
        "--syn-obj-path",
        type=str,
        default=None,
        help=(
            "Path to load image of synthetic object from (default: auto-"
            "generated from dataset and obj_class)."
        ),
    )
    parser.add_argument(
        "--syn-obj-width-px",
        type=int,
        default=None,
        help="Object width in pixels (default: 0.1 * img_size).",
    )
    parser.add_argument(
        "--syn-desired-fnr",
        type=float,
        default=None,
        help=(
            "If conf_thres is not specified, it will be computed such that "
            "prediction achieves syn_desired_fnr. We use this method to get "
            "conf_thres for synthetic data since we cannot compute FPR."
        ),
    )
    parser.add_argument(
        "--syn-rotate",
        type=float,
        default=15,
        help="Max rotation degrees for synthetic sign (default: 15).",
    )
    parser.add_argument(
        "--syn-scale",
        type=float,
        default=None,
        help=(
            "Scaling transform when evaluating on synthetic signs; Must be "
            "larger than 1; Scale is set to (1/syn_scale, syn_scale); "
            "(default: None).",
        ),
    )
    parser.add_argument(
        "--syn-3d-dist",
        type=float,
        default=None,
        help=(
            "Perspective transform distortion for synthetic sign; Override "
            "affine transform params (rorate, scale, translate); "
            "Not recommended because translation is very small (default: None)."
        ),
    )
    parser.add_argument(
        "--syn-colorjitter",
        type=float,
        default=None,
        help=(
            "Color jitter intensity for brightness, contrast, saturation "
            "for synthetic sign (default: None)."
        ),
    )

    # =========================== Attack arguments ========================== #
    parser.add_argument(
        "--attack-type",
        type=str,
        default="none",
        help=(
            'Attack evaluation to run: "none" (default), "load", "per-sign", '
            '"random", "debug".'
        ),
    )
    parser.add_argument(
        "--adv-patch-path",
        type=str,
        default=None,
        help="Path to adv patch and mask to load.",
    )
    parser.add_argument(
        "--patch-size",
        type=str,
        default="10x10",
        help=(
            "Specify size of adversarial patch to generate in inches "
            "(HxW); height and width separated by 'x'."
        ),
    )
    parser.add_argument(
        "--obj-class",
        type=int,
        default=-1,
        help="class of object to attack (-1: all classes)",
    )
    parser.add_argument(
        "--tgt-csv-filepath",
        type=str,
        default="",
        help="path to csv which contains target points for transform.",
    )
    parser.add_argument(
        "--attack-config-path",
        type=str,
        default="",
        help="Path to YAML file with attack configs.",
    )
    parser.add_argument(
        "--split-file-path",
        type=str,
        default="",
        help="path to a text file containing image filenames",
    )
    parser.add_argument(
        "--reap-geo-method",
        type=str,
        default="perspective",
        help=(
            'Transform type to use on patch: "perspective" (default), '
            '"translate_scale". This can be different from '
            "patch generation specified in attack config."
        ),
    )
    parser.add_argument(
        "--reap-relight-method",
        type=str,
        default="color_transfer",
        help=(
            'Relight transform method on patch: "color_transfer" (default), '
            '"percentile", "polynomial".'
        ),
    )
    parser.add_argument(
        "--reap-relight-polynomial-degree",
        type=int,
        default=1,
        help=(
            "Degree of polynomial for polynomial relighting (when "
            'reap_relight_method == "polynomial").'
        ),
    )
    parser.add_argument(
        "--reap-relight-percentile",
        type=int,
        default=1,
        help=(
            "Percentile of pixels with highest error to drop when computing "
            'relight coeffs. Used when reap_relight_method is "polynomial" or '
            '"percentile".'
        ),
    )
    parser.add_argument(
        "--options",
        type=str,
        default="",
        help=(
            "Custom attack options; will overwrite config file. Use '.' to "
            'impose hierarchy and space to separate options, e.g., -o "'
            'common.patch_dim=64 rp2.num_steps=1000".'
        ),
    )

    # TODO(YOLO): make general, not only detectron
    # parser.add_argument(
    #     "--iou-thres",
    #     type=float,
    #     default=0.5,
    #     help=(
    #         "IoU threshold to consider a match between ground-truth and "
    #         "predicted bbox."
    #     ),
    # )
    parser.add_argument(
        "--use-mixed-batch",
        action="store_true",
        help="Use mixed batch for adversarial training.",
    )
    parser.add_argument(
        "--skip-bg",
        action="store_true",
        help="Skip images with only background or 'other' classes.",
    )

    # ===================== Patch generation arguments ====================== #
    parser.add_argument(
        "--save-images", action="store_true", help="Save generated patch"
    )

    if is_detectron:
        # Detectron-specific arguments
        parser.add_argument(
            "--dt-config-file",
            type=str,
            help="Path to config file for detectron",
        )
    else:
        # ========================= YOLO arguments ========================== #
        # DEPRECATED
        # parser.add_argument(
        #     "--data",
        #     type=str,
        #     default=root / "data/coco128.yaml",
        #     help="dataset.yaml path",
        # )
        # parser.add_argument(
        #     "--imgsz",
        #     "--img",
        #     "--img-size",
        #     type=int,
        #     default=640,
        #     help="inference size (pixels)",
        # )
        # parser.add_argument(
        #     "--iou-thres", type=float, default=0.6, help="NMS IoU threshold"
        # )
        # parser.add_argument(
        #     "--task", default="val", help="train, val, test, speed or study"
        # )
        # parser.add_argument(
        #     "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
        # )
        # parser.add_argument(
        #     "--single-cls",
        #     action="store_true",
        #     help="treat as single-class dataset",
        # )
        # parser.add_argument(
        #     "--augment", action="store_true", help="augmented inference"
        # )
        # parser.add_argument(
        #     "--save-txt", action="store_true", help="save results to *.txt"
        # )
        # parser.add_argument(
        #     "--save-hybrid",
        #     action="store_true",
        #     help="save label+prediction hybrid results to *.txt",
        # )
        # parser.add_argument(
        #     "--save-conf",
        #     action="store_true",
        #     help="save confidences in --save-txt labels",
        # )
        # parser.add_argument(
        #     "--save-json",
        #     action="store_true",
        #     help="save a COCO-JSON results file",
        # )
        # parser.add_argument(
        #     "--project", default=root / "runs/val", help="save to project/name"
        # )
        # parser.add_argument(
        #     "--exist-ok",
        #     action="store_true",
        #     help="existing project/name ok, do not increment",
        # )
        # parser.add_argument(
        #     "--half",
        #     action="store_true",
        #     help="use FP16 half-precision inference",
        # )
        # parser.add_argument(
        #     "--dnn",
        #     action="store_true",
        #     help="use OpenCV DNN for ONNX inference",
        # )
        pass

    # ============================== Plot / log ============================= #
    parser.add_argument(
        "--save-exp-metrics",
        action="store_true",
        help="save metrics for this experiment to dataframe",
    )
    parser.add_argument(
        "--plot-single-images",
        action="store_true",
        help=(
            "Save single images in a folder instead of batch images in a "
            "single plot"
        ),
    )
    parser.add_argument(
        "--plot-class-examples",
        type=str,
        default="",
        nargs="*",
        help=(
            "Save single images containing individual classes in different "
            "folders."
        ),
    )
    parser.add_argument(
        "--metrics-confidence-threshold",
        type=float,
        default=None,
        help="confidence threshold",
    )
    parser.add_argument(
        "--plot-fp",
        action="store_true",
        help="save images containing false positives",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=0,
        help="Number of images to visualize (default: 0).",
    )
    parser.add_argument(
        "--vis-conf-thres",
        type=int,
        default=0.5,
        help=(
            "Min. confidence threshold of predictions to draw bbox for during "
            "visualization (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--vis-show-bbox",
        action="store_true",
        help="Whether to draw bbox in visualized images.",
    )

    args = parser.parse_args()

    if args.exp_config_file:
        # Load config from YAML file
        with open(args.exp_config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        # These configs are only set as default so command line args overwrite
        # config file.
        parser.set_defaults(**config["base"])
        args = parser.parse_args()
        attack_config = config["attack"]
        args = vars(args)
        config = {"base": args, "attack": attack_config}
    else:
        args = vars(args)
        config = {"base": args, "attack": {}}

    # Update attack config through command line args
    for opt in args["options"].split():
        tokens: List[str] = opt.split("=")
        if len(tokens) != 2:
            raise ValueError(
                "Attack options must be a key-value pair separated by '=', but "
                f"found {opt}."
            )
        params: List[str] = tokens[0].split(".")
        parent: Dict[str, Any] = config
        for i, param in enumerate(params):
            if param not in parent:
                raise ValueError(
                    f"This param ({tokens[0]}) is not defined in attack config "
                    "file. This is likely not a valid param."
                )
            if i < len(params) - 1:
                parent = parent[param]
                continue
            try:
                val = ast.literal_eval(tokens[1])
            except (ValueError, SyntaxError):
                val = tokens[1]
            parent[param] = val

    assert "base" in config and "attack" in config
    _update_dataset_name(config, is_train)
    _verify_base_config(config["base"], is_detectron)

    # Update config and fill with auto-generated params
    _update_img_size(config)
    _update_split_file(config, is_gen_patch)
    _update_syn_obj_path(config)
    _update_syn_obj_size(config)
    _update_attack_transforms(config)
    _update_save_dir(config, is_detectron=is_detectron, is_train=is_train)
    _update_result_dir(config)
    _update_conf_thres(config, is_train=is_train)

    config["base"]["verbosity"] = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG,
    }[config["base"]["verbosity"]]
    if config["base"]["debug"]:
        config["base"]["verbosity"] = logging.DEBUG

    return config


def _verify_base_config(config_base: Dict[str, Any], is_detectron: bool):
    """Some manual verifications of config.

    Args:
        config_base: Eval config dict.
        is_detectron: Whether detectron is used.
    """
    allowed_interp = INTERPS
    allowed_attack_types = ("none", "load", "per-sign", "random", "debug")
    allowed_yolo_models = ("yolov5", "yolor")
    allowed_datasets = ("reap", "synthetic", "mtsd", "mapillary")
    dataset: str = config_base["dataset"]

    if config_base["interp"] not in allowed_interp:
        raise ValueError(
            f"interp must be in {allowed_interp}, but it is "
            f'{config_base["interp"]}!'
        )

    if config_base["attack_type"] not in allowed_attack_types:
        raise ValueError(
            f"attack_type must be in {allowed_attack_types}, but it is "
            f'{config_base["attack_type"]}!'
        )

    # Verify dataset
    if dataset.split("-")[0] not in allowed_datasets:
        raise ValueError(
            f"dataset must be in {allowed_datasets}, but it is {dataset}!"
        )

    # Verify obj_class arg
    obj_class = config_base["obj_class"]
    max_cls = len(DATASET_METADATA[dataset]["class_name"]) - 1
    if not -1 <= obj_class <= max_cls:
        raise ValueError(
            f"Target object class {obj_class} is not between 0 and {max_cls}!"
        )

    if not is_detectron:
        if config_base["model_name"] not in allowed_yolo_models:
            raise ValueError(
                f"model_name (YOLO) must be in {allowed_yolo_models}, but it "
                f'is {config_base["model_name"]}!'
            )


def _update_conf_thres(
    config: Dict[str, Dict[str, Any]], is_train: bool = False
) -> None:
    config_base = config["base"]
    dataset = config_base["dataset"]
    if config_base["compute_conf_thres"]:
        config_base["conf_thres"] = None
    if (
        config_base["compute_conf_thres"]
        and dataset == "synthetic"
        and config_base["syn_desired_fnr"] is None
    ):
        raise ValueError(
            "For synthetic data, if compute_conf_thres is True, "
            "syn_desired_fnr must be specified."
        )
    # Load conf_thres from metadata
    if (
        not config_base["compute_conf_thres"]
        and config_base["conf_thres"] is None
        and config_base["weights"] is not None
        and not is_train
    ):
        weights_path = pathlib.Path(config_base["weights"])
        metadata_dir = weights_path.parent / "metadata.pkl"
        if not metadata_dir.is_file():
            raise FileNotFoundError(
                "Metadata is not found in the default location "
                f"{str(metadata_dir)}! when loading conf_thres. Please place "
                "metadata.pkl there, or set compute_conf_thres to False and "
                "specify conf_thres."
            )
        with metadata_dir.open("rb") as file:
            base_metadata = pickle.load(file)
        if dataset in base_metadata:
            # For backward compatibility
            conf_thres = base_metadata[dataset]["conf_thres"]
        else:
            conf_thres = base_metadata[weights_path.name][dataset]["conf_thres"]
        if isinstance(conf_thres, np.ndarray):
            conf_thres = conf_thres.tolist()
        config_base["conf_thres"] = conf_thres


def _update_dataset_name(
    config: Dict[str, Dict[str, Any]], is_train: bool = False
) -> None:
    """Update dataset and dataset_split in config_base.

    Expect dataset to have 1-3 tokens separated by "-" (hyphen). The first token
    must be base name (e.g., "reap", "mtsd", etc.). The other two can be split
    (e.g., "train", "test", or "val") or label indication (called "color" for
    now). color is either "color" or "no_color" (default) and is only applicable
    to mtsd and mapillary.
    """
    config_base: Dict[str, Any] = config["base"]
    dataset: str = config_base["dataset"]
    # <BASE_DATASET>-<MODIFIER>-<SPLIT>: e.g., mtsd-no_color-train
    _, use_color, _, _, _, _, split = parse_dataset_name(dataset)
    dataset_no_split = dataset
    if split is not None:
        dataset_no_split = "-".join(dataset.split("-")[:-1])

    if any(name in dataset for name in ("reap", "synthetic")):
        config_base["use_color"] = False
        config_base["synthetic"] = "synthetic" in dataset
        # REAP benchmark only uses annotated signs. Synthetic data use both.
        config_base["annotated_signs_only"] = not config_base["synthetic"]
        split = "combined"
    else:
        if split not in ("train", "val", "test", "combined"):
            split = "train" if is_train else "test"
        config_base["synthetic"] = False
        config_base["use_color"] = use_color
    config_base["dataset"] = dataset_no_split
    config_base["dataset_split"] = split

    if config_base["train_dataset"] is None:
        config_base["train_dataset"] = f"{dataset_no_split}_train"


def _update_split_file(
    config: Dict[str, Dict[str, Any]], is_gen_patch: bool
) -> None:
    config_base = config["base"]
    split_file_path: Optional[str] = config_base["split_file_path"]
    dataset: str = config_base["dataset"]
    num_bg: int = config["attack"]["common"]["num_bg"]
    if split_file_path is None:
        logger.info("split_file_path is not specified.")
        return

    if os.path.isfile(split_file_path):
        logger.info("split_file_path is specified as %s.", split_file_path)
        return

    # If split_file is dir, we search inside that dir to find valid txt file
    # given obj_class.
    split_file_dir = pathlib.Path(split_file_path)
    if not split_file_dir.is_dir():
        raise FileNotFoundError(
            f"split_file_path ({split_file_dir}) is not dir."
        )

    # Try to find split file in given dir
    split: str = "attack" if is_gen_patch else "test"
    if config_base["obj_class"] < 0:
        default_filename: str = "all.txt"
    else:
        class_name: str = DATASET_METADATA[dataset]["class_name"][
            config_base["obj_class"]
        ]
        default_filename: str = f"{class_name}_{split}.txt"

    split_file_path: pathlib.Path = split_file_dir / default_filename
    if split_file_path.is_file():
        logger.info("Using split_file_path: %s.", split_file_path)
        config_base["split_file_path"] = split_file_path
        return

    # Try to automatically generate correct path to split file
    split_file_path = (
        split_file_dir / dataset / f"bg_{num_bg}" / default_filename
    )
    if not split_file_path.is_file():
        raise FileNotFoundError(
            "split_file_path is dir, but cannot find a valid split file at "
            f"{str(split_file_path)} (auto-generated)."
        )

    split_file_path = str(split_file_path)
    logger.info("Using auto-generated split_file_path: %s.", split_file_path)
    config_base["split_file_path"] = split_file_path


def _update_syn_obj_path(config: Dict[str, Dict[str, Any]]) -> None:
    """Update path to load synthetic object from.

    Args:
        config: Config dict.
    """
    config_base = config["base"]
    if config_base["syn_obj_path"] is not None:
        return
    dataset = config_base["dataset"]
    obj_class = config_base["obj_class"]
    if obj_class >= 0:
        class_name = DATASET_METADATA[dataset]["class_name"][obj_class]
        config_base["syn_obj_path"] = os.path.join(
            DEFAULT_SYN_OBJ_DIR, dataset, f"{class_name}.png"
        )


def _update_img_size(config: Dict[str, Dict[str, Any]]) -> None:
    config_base = config["base"]
    img_size = config_base["padded_imgsz"]
    img_size = tuple(int(s) for s in img_size.split(","))
    config_base["img_size"] = img_size
    config_base["padded_imgsz"] = img_size
    # Verify given image size
    if len(img_size) != 2 or not all(isinstance(s, int) for s in img_size):
        raise ValueError(f"padded_imgsz has wrong format: {img_size}")


def _update_syn_obj_size(config: Dict[str, Dict[str, Any]]) -> None:
    """Update syn_obj_size_px and syn_obj_size_mm params in eval config.

    Args:
        config: Config dict.

    Raises:
        ValueError: syn_obj_width_px is not int.
    """
    config_base = config["base"]
    # Get real object size of that class
    dataset = config_base["dataset"]
    obj_dim_dict = DATASET_METADATA[dataset]
    obj_class = config_base["obj_class"]

    hw_ratio = 1.0
    if obj_class >= 0:
        hw_ratio = obj_dim_dict["hw_ratio"][obj_class]
        config_base["obj_size_mm"] = obj_dim_dict["size_mm"][obj_class]

    if not config_base["synthetic"]:
        # For attack using real images, we still need to specify obj_size_px to
        # come up with the correctly-sized mask so obj_size is set to patch_dim
        # from attack config.
        if "attack" in config:
            patch_dim = config["attack"]["common"]["patch_dim"]
            config_base["obj_size_px"] = (
                round(patch_dim * hw_ratio),
                patch_dim,
            )
        return

    obj_width_px = config_base["syn_obj_width_px"]
    if not isinstance(obj_width_px, int):
        raise ValueError(f"syn_obj_width_px must be int ({obj_width_px})!")

    config_base["obj_size_px"] = (
        round(obj_width_px * hw_ratio),
        obj_width_px,
    )


def _update_save_dir(
    config: Dict[str, Dict[str, Any]],
    is_detectron: bool = True,
    is_train: bool = False,
) -> None:
    """Create folder for saving eval results and set save_dir in config.

    Args:
        config: Config dict.

    Raises:
        ValueError: Invalid obj_class.
    """
    config_base = config["base"]
    config_atk = config["attack"]["common"]
    synthetic = config_base["synthetic"]

    # Automatically generate experiment name
    token_list: List[str] = []
    token_list.append(config_base["dataset"])
    token_list.append(config_base["model_name"])
    token_list.append(config_base["attack_type"])

    if config_base["attack_type"] != "none":
        token_list.append(config_base["patch_size"])

    if config_base["attack_type"] in ("load", "per-sign"):

        # Gather dataset-specific transform params
        if synthetic:
            for param in _TRANSFORM_PARAMS:
                if "syn" in param:
                    token_list.append(str(config_atk[param]))
        token_list.append(config_atk["reap_relight_method"])
        if config_atk["reap_relight_method"] == "polynomial":
            token_list.append(config_atk["reap_relight_polynomial_degree"])
            token_list.append(config_atk["reap_relight_percentile"])
        elif config_atk["reap_relight_method"] == "percentile":
            token_list.append(config_atk["reap_relight_percentile"])
        if config_atk["reap_geo_method"] != "perspective":
            token_list.append(config_atk["reap_geo_method"])

        token_list.append(f"pd{config_atk['patch_dim']}")
        token_list.append(f"bg{config_atk['num_bg']}")

        # Geometric transform augmentation
        if config_atk["aug_prob_geo"] > 0:
            aug_3d_dist = config_atk["aug_3d_dist"]
            if aug_3d_dist is not None and aug_3d_dist > 0:
                params = ["aug_prob_geo", "aug_3d_dist"]
            else:
                params = [
                    "aug_prob_geo",
                    "aug_rotate",
                    "aug_translate",
                    "aug_scale",
                ]
            aug_geo_tokens = []
            for param in params:
                aug_geo_tokens.append(str(config_atk[param]))
            token_list.append("auggeo" + "_".join(aug_geo_tokens))

        # Lighting augmentation (color jitter)
        aug_cj = config_atk["aug_prob_colorjitter"]
        if aug_cj is not None and aug_cj > 0:
            light = (
                f"augcj{config_atk['aug_prob_colorjitter']}_"
                f"{config_atk['aug_colorjitter']}"
            )
            token_list.append(light)

        # Background image augmentation params
        img_aug_prob_geo: Optional[float] = config_atk["img_aug_prob_geo"]
        if img_aug_prob_geo is not None and img_aug_prob_geo > 0:
            token_list.append(f"augimg{img_aug_prob_geo}")

        attack_name: str = config_atk["attack_name"]
        atk_params: Dict[str, Any] = config["attack"][attack_name]
        # Sort attack-specific params so we can print all at once
        atk_params = dict(sorted(atk_params.items()))
        atk_params_list: List[str] = [
            str(v) for v in atk_params.values() if not isinstance(v, dict)
        ]
        token_list.append(attack_name + "_".join(atk_params_list))

        if config_base["use_mixed_batch"]:
            token_list.append("mix")

    # Append custom name at the end
    token_list = [str(t) for t in token_list]
    exp_name = "-".join(token_list)
    if config_base["name"] is not None:
        name_from_cfg: str = str(config_base["name"])
        if name_from_cfg.startswith("_"):
            exp_name += name_from_cfg
        else:
            exp_name = name_from_cfg
    exp_name = f"train_{exp_name}" if is_train else f"test_{exp_name}"
    config_base["name"] = exp_name

    obj_class: int = config_base["obj_class"]
    if not is_train:
        class_name = (
            DATASET_METADATA[config_base["dataset"]]["class_name"][obj_class]
            if obj_class >= 0
            else "all"
        )
        save_dir = os.path.join(config_base["base_dir"], exp_name, class_name)
    else:
        save_dir = os.path.join(config_base["base_dir"], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    config_base["save_dir"] = save_dir


def _update_attack_transforms(config: Dict[str, Dict[str, Any]]) -> None:
    config_base: Dict[str, Any] = config["base"]
    config_atk: Dict[str, Any] = config["attack"]["common"]
    for param in _TRANSFORM_PARAMS:
        if param not in config_atk or config_atk[param] is None:
            config_atk[param] = config_base[param]


def _update_result_dir(config: Dict[str, Dict[str, Any]]) -> None:
    config_base = config["base"]
    result_dir = config_base["attack_type"] + (
        "_syn" if config_base["synthetic"] else ""
    )
    result_dir = os.path.join(config_base["save_dir"], result_dir)
    os.makedirs(result_dir, exist_ok=True)
    config_base["result_dir"] = result_dir


def setup_detectron_cfg(
    config: Dict[str, Dict[str, Any]],
    is_train: bool = False,
) -> detectron2.config.CfgNode:
    """Create configs and perform basic setups."""
    config_base = config["base"]
    dataset: str = config_base["dataset"]
    split: str = config_base["dataset_split"]
    num_classes = len(DATASET_METADATA[dataset]["class_name"])

    # Set default path to load adversarial patch
    if not config_base["adv_patch_path"]:
        config_base["adv_patch_path"] = os.path.join(
            config_base["save_dir"], "adv_patch.pkl"
        )

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


def parse_dataset_name(dataset_name: str) -> list[str, bool, int]:
    """Parse dataset name to get base dataset name and modifiers."""
    base_dataset = dataset_name.split("-")[0]
    dataset_modifiers: list[str] = []
    if "-" in dataset_name:
        dataset_modifiers = dataset_name.split("-")[1:]

    if base_dataset not in ("synthetic", "reap"):
        # Whether sign color is used for labels. Defaults to False
        use_color = "color" in dataset_modifiers
        # Whether to use original MTSD labels instead of REAP annotations
        use_orig_labels = "orig" in dataset_modifiers
        # Whether to skip images with no object of interest
        skip_bg_only = "skipbg" in dataset_modifiers
    else:
        use_color = False
        use_orig_labels = False
        skip_bg_only = base_dataset == "reap"

    # Whether to ignore background class (last class index) and not include it
    # in dataset dict and targets
    ignore_bg_class = "nobg" in dataset_modifiers

    # Get num classes like mtsd-100, reap-100, etc.
    num_classes = None
    if "100" in dataset_modifiers:
        num_classes = 100

    # Get split
    split = None
    for split_name in ("train", "val", "test", "combined"):
        if split_name in dataset_modifiers:
            split = split_name
            break

    return (
        base_dataset,
        use_color,
        use_orig_labels,
        ignore_bg_class,
        skip_bg_only,
        num_classes,
        split,
    )


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
