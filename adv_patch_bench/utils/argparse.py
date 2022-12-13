"""Define argument and config parsing."""

import argparse
import ast
import os
import pathlib
from typing import Any, Dict, List, Optional, Union

import detectron2
import yaml
from detectron2.engine import default_argument_parser, default_setup

from hparams import (
    DEFAULT_SYN_OBJ_DIR,
    INTERPS,
    LABEL_LIST,
    NUM_CLASSES,
    OBJ_DIM_DICT,
)

_TRANSFORM_PARAMS: List[str] = [
    "interp",
    "reap_transform_mode",
    "reap_use_relight",
    "syn_obj_width_px",
    "syn_rotate",
    "syn_scale",
    "syn_translate",
    "syn_colorjitter",
    "syn_3d_dist",
]


def eval_args_parser(
    is_detectron: bool,
    root: Optional[str] = None,
    is_gen_patch: bool = False,
) -> Dict[str, Any]:
    """Setup argparse for evaluation.

    TODO(enhancement): Improve argument documentation.

    Args:
        is_detectron: Whether we are evaluating dectectron model.
        root: Path to root or base directory. Defaults to None.

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
    parser.add_argument("--dataset", type=str, required=False)
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
    # TODO: DEPRECATE this for YOLO
    # parser.add_argument(
    #     "--synthetic",
    #     action="store_true",
    #     help="evaluate with pasted synthetic signs",
    # )
    parser.add_argument(
        "--name", type=str, default=None, help="save to project/name"
    )
    parser.add_argument("--verbose", action="store_true")
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
        "--patch-size-inch",
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
    # TODO: This can be removed? This is specified through reap-transform-mode
    # parser.add_argument(
    #     "--no-patch-transform",
    #     action="store_true",
    #     help=(
    #         "If True, do not apply patch to signs using 3D-transform. "
    #         "Patch will directly face camera."
    #     ),
    # )
    parser.add_argument(
        "--reap-transform-mode",
        type=str,
        default="perspective",
        help=(
            "transform type to use on patch during evaluation: perspective "
            "(default), affine, translate_scale. This can be different from "
            "patch generation specified in attack config."
        ),
    )
    parser.add_argument(
        "-a",
        "--attack-options",
        type=str,
        default="",
        help=(
            "Custom attack options; will overwrite config file. Use '.' to "
            'impose hierarchy and space to separate options, e.g., -a "'
            'common.patch_dim=64 rp2.num_steps=1000".'
        ),
    )
    parser.add_argument(
        "--reap-use-relight",
        action="store_true",
        help="If True, apply relighting transform to patch.",
    )
    # TODO(deprecated):
    # parser.add_argument(
    #     "--min-area",
    #     type=float,
    #     default=0,
    #     help=(
    #         "(DEPRECATED) Minimum area for labels. if a label has area > "
    #         "min_area, predictions correspoing to this target will be discarded"
    #     ),
    # )
    # TODO(deprecated): DEPRECATED: is this stil used?
    # parser.add_argument(
    #     "--min-pred-area",
    #     type=float,
    #     default=0,
    #     help=(
    #         "Minimum area for predictions. If predicion has area < min_area and"
    #         " that prediction is not matched by any gt, it will be discarded."
    #     ),
    # )

    # TODO(YOLO): make general, not only detectron
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help=(
            "IoU threshold to consider a match between ground-truth and "
            "predicted bbox."
        ),
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
        parser.add_argument(
            "--data",
            type=str,
            default=root / "data/coco128.yaml",
            help="dataset.yaml path",
        )

        parser.add_argument(
            "--batch-size", type=int, default=32, help="batch size"
        )
        parser.add_argument(
            "--imgsz",
            "--img",
            "--img-size",
            type=int,
            default=640,
            help="inference size (pixels)",
        )
        parser.add_argument(
            "--iou-thres", type=float, default=0.6, help="NMS IoU threshold"
        )
        parser.add_argument(
            "--task", default="val", help="train, val, test, speed or study"
        )
        parser.add_argument(
            "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
        )
        parser.add_argument(
            "--single-cls",
            action="store_true",
            help="treat as single-class dataset",
        )
        parser.add_argument(
            "--augment", action="store_true", help="augmented inference"
        )
        parser.add_argument(
            "--save-txt", action="store_true", help="save results to *.txt"
        )
        parser.add_argument(
            "--save-hybrid",
            action="store_true",
            help="save label+prediction hybrid results to *.txt",
        )
        parser.add_argument(
            "--save-conf",
            action="store_true",
            help="save confidences in --save-txt labels",
        )
        parser.add_argument(
            "--save-json",
            action="store_true",
            help="save a COCO-JSON results file",
        )
        parser.add_argument(
            "--project", default=root / "runs/val", help="save to project/name"
        )
        parser.add_argument(
            "--exist-ok",
            action="store_true",
            help="existing project/name ok, do not increment",
        )
        parser.add_argument(
            "--half",
            action="store_true",
            help="use FP16 half-precision inference",
        )
        parser.add_argument(
            "--dnn",
            action="store_true",
            help="use OpenCV DNN for ONNX inference",
        )

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

    # TODO(deprecated): remove in the future
    # parser.add_argument(
    #     "--other-class-confidence-threshold",
    #     type=float,
    #     default=0,
    #     help=(
    #         "confidence threshold at which other labels are changed if there "
    #         "is a match with a prediction"
    #     ),
    # )

    args = parser.parse_args()

    if args.exp_config_file:
        # Load config from YAML file
        with open(args.exp_config_file, "r") as f:
            config = yaml.safe_load(f)
        # These configs are only set as default so command line args overwrite
        # config file.
        parser.set_defaults(**config["eval"])
        args = parser.parse_args()
        attack_config = config["attack"]
        args = vars(args)
        config = {"eval": args, "attack": attack_config}
    else:
        args = vars(args)
        config = {"eval": args, "attack": {}}

    # Update attack config through command line args
    for opt in args["attack_options"].split():
        tokens: List[str] = opt.split("=")
        if len(tokens) != 2:
            raise ValueError(
                "Attack options must be a key-value pair separated by '=', but "
                f"found {opt}."
            )
        params: List[str] = tokens[0].split(".")
        parent: Dict[str, Any] = config["attack"]
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
            except ValueError:
                val = tokens[1]
            parent[param] = val

    assert "eval" in config and "attack" in config
    _update_dataset_name(config)
    _verify_eval_config(config["eval"], is_detectron)

    # Update config and fill with auto-generated params
    _update_img_size(config)
    _update_split_file(config, is_gen_patch)
    _update_syn_obj_path(config)
    _update_syn_obj_size(config)
    _update_patch_size(config)
    _update_attack_transforms(config)
    _update_save_dir(config, is_detectron)
    _update_result_dir(config)

    return config


def _verify_eval_config(config_eval: Dict[str, Any], is_detectron: bool):
    """Some manual verifications of config.

    Args:
        config_eval: Eval config dict.
        is_detectron: Whether detectron is used.
    """
    allowed_interp = INTERPS
    allowed_attack_types = ("none", "load", "per-sign", "random", "debug")
    allowed_yolo_models = ("yolov5", "yolor")
    allowed_datasets = ("reap", "synthetic", "mtsd", "mapillary")
    dataset: str = config_eval["dataset"]

    if config_eval["interp"] not in allowed_interp:
        raise ValueError(
            f"interp must be in {allowed_interp}, but it is "
            f'{config_eval["interp"]}!'
        )

    if config_eval["attack_type"] not in allowed_attack_types:
        raise ValueError(
            f"attack_type must be in {allowed_attack_types}, but it is "
            f'{config_eval["attack_type"]}!'
        )

    # Verify dataset
    if dataset.split("_")[0] not in allowed_datasets:
        raise ValueError(
            f"dataset must be in {allowed_datasets}, but it is {dataset}!"
        )

    # Verify obj_class arg
    obj_class = config_eval["obj_class"]
    max_cls = NUM_CLASSES[dataset] - 1
    if not 0 <= obj_class <= max_cls:
        raise ValueError(
            f"Target object class {obj_class} is not between 0 and {max_cls}!"
        )

    if not is_detectron:
        if config_eval["model_name"] not in allowed_yolo_models:
            raise ValueError(
                f"model_name (YOLO) must be in {allowed_yolo_models}, but it "
                f'is {config_eval["model_name"]}!'
            )


def _update_dataset_name(config: Dict[str, Dict[str, Any]]) -> List[str]:
    """Update dataset and dataset_split in config_eval."""
    config_eval: Dict[str, Any] = config["eval"]
    dataset: str = config_eval["dataset"]
    tokens: List[str] = dataset.split("-")
    if dataset in ("reap", "synthetic"):
        config_eval["use_color"] = False
        config_eval["synthetic"] = dataset == "synthetic"
        # REAP benchmark only uses annotated signs. Synthetic data use both.
        config_eval["annotated_signs_only"] = not config_eval["synthetic"]
        split = "combined"
    else:
        assert len(tokens) in (2, 3), f"Invalid dataset: {dataset}!"
        if len(tokens) == 2:
            base_dataset, split = tokens
            color = "no_color"
        else:
            base_dataset, color, split = tokens
        dataset: str = f"{base_dataset}_{color}"
        config_eval["synthetic"] = False
        config_eval["use_color"] = "color" == color
    config_eval["dataset"] = dataset
    config_eval["dataset_split"] = split


def _update_split_file(
    config: Dict[str, Dict[str, Any]], is_gen_patch: bool
) -> None:
    config_eval = config["eval"]
    split_file_path: Optional[str] = config_eval["split_file_path"]
    dataset: str = config_eval["dataset"]
    num_bg: int = config["attack"]["common"]["num_bg"]
    if split_file_path is None:
        print("split_file_path is not specified.")
        return

    if os.path.isfile(split_file_path):
        print(f"split_file_path is specified as {split_file_path}.")
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
    class_name: str = LABEL_LIST[dataset][config_eval["obj_class"]]
    default_filename: str = f"{class_name}_{split}.txt"
    split_file_path: pathlib.Path = split_file_dir / default_filename
    if split_file_path.is_file():
        print(f"Using split_file_path: {split_file_path}.")
        config_eval["split_file_path"] = split_file_path
        return

    # Try to automatically generate correct path to split file
    split_file_path = (
        split_file_dir / dataset / f"bg{num_bg}" / default_filename
    )
    if not split_file_path.is_file():
        raise FileNotFoundError(
            "split_file_path is dir, but cannot find a valid split file at "
            f"{str(split_file_path)} (auto-generated)."
        )

    split_file_path = str(split_file_path)
    print(f"Using auto-generated split_file_path: {split_file_path}.")
    config_eval["split_file_path"] = split_file_path


def _update_syn_obj_path(config: Dict[str, Dict[str, Any]]) -> None:
    """Update path to load synthetic object from.

    Args:
        config: Config dict.
    """
    config_eval = config["eval"]
    if config_eval["syn_obj_path"] is not None:
        return
    dataset = config_eval["dataset"]
    obj_class = config_eval["obj_class"]
    class_name = LABEL_LIST[dataset][obj_class]
    config_eval["syn_obj_path"] = os.path.join(
        DEFAULT_SYN_OBJ_DIR, dataset, f"{class_name}.png"
    )


def _update_img_size(config: Dict[str, Dict[str, Any]]) -> None:
    config_eval = config["eval"]
    img_size = config_eval["padded_imgsz"]
    img_size = tuple([int(s) for s in img_size.split(",")])
    config_eval["img_size"] = img_size
    config_eval["padded_imgsz"] = img_size
    # Verify given image size
    if len(img_size) != 2 or not all([isinstance(s, int) for s in img_size]):
        raise ValueError(f"padded_imgsz has wrong format: {img_size}")


def _update_syn_obj_size(config: Dict[str, Dict[str, Any]]) -> None:
    """Update syn_obj_size_px and syn_obj_size_mm params in eval config.

    Args:
        config: Config dict.

    Raises:
        ValueError: syn_obj_width_px is not int.
    """
    config_eval = config["eval"]
    # Get real object size of that class
    dataset = config_eval["dataset"]
    obj_dim_dict = OBJ_DIM_DICT[dataset]
    obj_class = config_eval["obj_class"]
    hw_ratio = obj_dim_dict["hw_ratio"][obj_class]
    config_eval["obj_size_mm"] = obj_dim_dict["size_mm"][obj_class]

    if not config_eval["synthetic"]:
        # For attack using real images, we still need to specify obj_size_px to
        # come up with the correctly-sized mask so obj_size is set to patch_dim
        # from attack config.
        if "attack" in config:
            patch_dim = config["attack"]["common"]["patch_dim"]
            config_eval["obj_size_px"] = (
                round(patch_dim * hw_ratio),
                patch_dim,
            )
        return

    obj_width_px = config_eval["syn_obj_width_px"]
    if not isinstance(obj_width_px, int):
        raise ValueError(f"syn_obj_width_px must be int ({obj_width_px})!")

    config_eval["obj_size_px"] = (
        round(obj_width_px * hw_ratio),
        obj_width_px,
    )


def _update_save_dir(
    config: Dict[str, Dict[str, Any]], is_detectron: bool
) -> None:
    """Create folder for saving eval results and set save_dir in config.

    Args:
        config: Config dict.

    Raises:
        ValueError: Invalid obj_class.
    """
    config_eval = config["eval"]
    config_atk = config["attack"]["common"]
    synthetic = config_eval["synthetic"]

    # Automatically generate experiment name
    token_list: List[str] = []
    token_list.append(config_eval["dataset"])
    token_list.append(config_eval["model_name"])
    token_list.append(config_eval["attack_type"])

    if config_eval["attack_type"] != "none":
        token_list.append(config_eval["patch_size_inch"])

    if config_eval["attack_type"] in ("load", "per-sign"):

        # Gather dataset-specific transform params
        if synthetic:
            for param in _TRANSFORM_PARAMS:
                if "syn" in param:
                    token_list.append(str(config_atk[param]))
        if not config_atk["reap_use_relight"]:
            token_list.append("nolight")
        if config_atk["reap_transform_mode"] != "perspective":
            token_list.append(config_atk["reap_transform_mode"])

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

        if is_detectron and "detectron" in atk_params:
            dt_params: Dict[str, Any] = atk_params["detectron"]
            dt_params = dict(sorted(dt_params.items()))
            dt_params_list: List[str] = [str(v) for v in dt_params.values()]
            token_list.append("dt" + "_".join(dt_params_list))

    # Append custom name at the end
    exp_name = "-".join(token_list)
    if config_eval["name"] is not None:
        name_from_cfg: str = str(config_eval["name"])
        if name_from_cfg.startswith("_"):
            exp_name += name_from_cfg
        else:
            exp_name = name_from_cfg
    config_eval["name"] = exp_name

    class_name = LABEL_LIST[config_eval["dataset"]][config_eval["obj_class"]]
    save_dir = os.path.join(config_eval["base_dir"], exp_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    config_eval["save_dir"] = save_dir


def _inch_to_mm(length_in_inch: Union[int, float]) -> float:
    return 25.4 * length_in_inch


def _update_patch_size(config: Dict[str, Dict[str, Any]]) -> None:
    config_eval = config["eval"]
    patch_size_inch: str = config_eval["patch_size_inch"]

    # patch_size has format <NUM_PATCH>_<HEIGHT>x<WIDTH>
    num_patches: int
    if len(patch_size_inch.split("_")) == 2:
        num_patches = int(patch_size_inch.split("_")[0])
    else:
        num_patches = 1
    if num_patches not in (1, 2):
        raise NotImplementedError(
            f"Only num_patches of 1 or 2 for now, but {num_patches} is given "
            f"({patch_size_inch})!"
        )

    patch_size = patch_size_inch.split("_")[-1].split("x")
    if not all(s.isnumeric() for s in patch_size):
        raise ValueError(f"Invalid patch size: {patch_size_inch}!")
    patch_size_inch = [int(s) for s in patch_size]
    patch_size_mm = [_inch_to_mm(s) for s in patch_size_inch]
    config_eval["patch_size_mm"] = (num_patches,) + tuple(patch_size_mm)


def _update_attack_transforms(config: Dict[str, Dict[str, Any]]) -> None:
    config_eval: Dict[str, Any] = config["eval"]
    config_atk: Dict[str, Any] = config["attack"]["common"]
    for param in _TRANSFORM_PARAMS:
        if param not in config_atk or config_atk[param] is None:
            config_atk[param] = config_eval[param]


def _update_result_dir(config: Dict[str, Dict[str, Any]]) -> None:
    config_eval = config["eval"]
    result_dir = config_eval["attack_type"] + (
        "_syn" if config_eval["synthetic"] else ""
    )
    result_dir = os.path.join(config_eval["save_dir"], result_dir)
    os.makedirs(result_dir, exist_ok=True)
    config_eval["result_dir"] = result_dir


def setup_detectron_test_args(
    config: Dict[str, Dict[str, Any]]
) -> detectron2.config.CfgNode:
    """Create configs and perform basic setups."""
    config_eval = config["eval"]
    dataset: str = config_eval["dataset"]
    split: str = config_eval["dataset_split"]

    # Set default path to load adversarial patch
    if not config_eval["adv_patch_path"]:
        config_eval["adv_patch_path"] = os.path.join(
            config_eval["save_dir"], "adv_patch.pkl"
        )

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(config_eval["config_file"])
    cfg.merge_from_list(config_eval["opts"])

    # Copy dataset from args
    cfg.DATASETS.TEST = (f"{dataset}_{split}",)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.CROP.ENABLED = False  # Turn off augmentation for testing
    cfg.DATALOADER.NUM_WORKERS = config_eval["workers"]
    cfg.eval_mode = config_eval["eval_mode"]
    cfg.obj_class = config_eval["obj_class"]
    # Assume that "other" class is always last
    cfg.other_catId = NUM_CLASSES[dataset] - 1
    config_eval["other_sign_class"] = cfg.other_catId
    cfg.conf_thres = config_eval["conf_thres"]

    # Set detectron image size from argument
    # Let img_size be (1536, 2048). This tries to set the smaller side of the
    # image to 2048, but the longer side cannot exceed 2048. The result of this
    # is that every image has its longer side (could be width or height) 2048.
    cfg.INPUT.MAX_SIZE_TEST = max(config_eval["img_size"])
    cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MAX_SIZE_TEST

    # Model config
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES[dataset]
    cfg.OUTPUT_DIR = config_eval["base_dir"]
    weight_path = config_eval["weights"]
    if isinstance(config_eval["weights"], list):
        weight_path = weight_path[0]
    assert isinstance(
        weight_path, str
    ), f"weight_path must be string, but it is {weight_path}!"
    cfg.MODEL.WEIGHTS = weight_path

    cfg.freeze()
    # Set cfg as global variable so we can avoid passing cfg around
    detectron2.config.set_global_cfg(cfg)
    config_eval.pop("config_file")  # Remove to avoid logging
    default_setup(cfg, argparse.Namespace(**config_eval))
    return cfg


def setup_yolo_test_args(config, other_sign_class):
    """Set up config for YOLO.

    # FIXME(YOLO): fix for yolo.

    Args:
        config: Config dict.
        other_sign_class: Class of "other" or background object.
    """
    config_eval = config["eval"]

    # Set YOLO data yaml file
    config_eval["data"] = config_eval["dataset"] + ".yaml"

    # Set to default value. This is different from conf_thres in detectron
    # args.conf_thres = 0.001
