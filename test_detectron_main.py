"""Test script for Detectron2 models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from packaging import version

# Calling subprocess.check_output() with python version 3.8.10 or lower will
# raise NotADirectoryError. When torch calls this to call hipconfig, it does
# not catch this exception but only FileNotFoundError or PermissionError.
# This hack makes sure that correct exception is raised.
if version.parse(sys.version.split()[0]) <= version.parse("3.8.10"):
    import subprocess

    def _hacky_subprocess_fix(*args, **kwargs):
        raise FileNotFoundError(
            "Hacky exception. If this interferes with your workflow, consider "
            "using python >= 3.8.10 or simply try to comment this out."
        )

    subprocess.check_output = _hacky_subprocess_fix

# pylint: disable=wrong-import-position
import detectron2
import torch

import adv_patch_bench.dataloaders.detectron.util as data_util
from adv_patch_bench.evaluators import detectron_evaluator
from adv_patch_bench.utils.argparse import reap_args_parser, setup_detectron_cfg
from hparams import LABEL_LIST

logger = logging.getLogger(__name__)
# This is to ignore a warning from detectron2/structures/keypoints.py:29
warnings.filterwarnings("ignore", category=UserWarning)

_EVAL_PARAMS = [
    "conf_thres",
    "dataset",
    "debug",
    "interp",
    "num_eval",
    "padded_imgsz",
    "patch_size_inch",
    "reap_transform_mode",
    "reap_use_relight",
    "seed",
    "syn_3d_dist",
    "syn_colorjitter",
    "syn_obj_width_px",
    "syn_rotate",
    "syn_scale",
    "syn_colorjitter",
    "syn_3d_dist",
    "model_name",
    "weights",
]
_EPS = np.spacing(1)
_NUM_SCORES = 1000


def _hash(obj: str) -> str:
    return hashlib.sha512(obj.encode("utf-8")).hexdigest()[:8]


def _hash_dict(config_dict: Dict[str, Any]) -> str:
    dict_str = json.dumps(config_dict, sort_keys=True)
    # Take first 8 characters of the hash since we prefer short file name
    return _hash(dict_str)


def _normalize_dict(
    orig_dict: Dict[str, Any], sep: str = "."
) -> Dict[str, Any]:
    [flat_dict] = pd.json_normalize(orig_dict, sep=sep).to_dict(
        orient="records"
    )
    flat_dict = {key: flat_dict[key] for key in sorted(flat_dict.keys())}
    return flat_dict


def _get_tp_fp_full(
    scores_full: list[list[float]], conf_thres: float | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true and false positives given scores and score threshold.

    Args:
        scores_full: List of list of scores. First axis is obj classes and
            second is number of IoU thresholds.
        conf_thres: Score threshold to consider a detection.

    Returns:
        True and false positives. Shape: [num_ious, num_classes].
    """
    num_classes = len(scores_full)
    num_ious = len(scores_full[0])
    tp_full = np.zeros((num_ious, num_classes))
    fp_full = np.zeros_like(tp_full)
    for i in range(num_ious):
        for k in range(num_classes):
            if isinstance(conf_thres, float):
                thres = conf_thres
            else:
                thres = conf_thres[k]
            tp_full[i, k] = np.sum(np.array(scores_full[k][i][0]) >= thres)
            fp_full[i, k] = np.sum(np.array(scores_full[k][i][1]) >= thres)
    return tp_full, fp_full


def _compute_conf_thres_syn(
    scores: np.ndarray, desired_fnr: float = 0.05
) -> float:
    """Compute confidence score threshold for synthetic data.

    Threshold is chosen such that FNR is equal to `desired_fnr`.

    Args:
        scores: Predicted class scores.
        desired_fnr: Desired FNR to achieve. Defaults to 0.05.

    Returns:
        Confidence score threshold.
    """
    logger.info(
        "Computing confidence score threshold for synthetic data to achieve "
        "FNR of %.3f...",
        desired_fnr,
    )
    scores_thres = np.linspace(0, 1, _NUM_SCORES)
    fnrs = (scores_thres[:, None] > scores[None]).mean(1)
    score_idx = np.where(fnrs < desired_fnr)[0][-1]
    # Round to 3 digits
    conf_score = scores_thres[score_idx].round(3)
    logger.info("Obtained confidence score threshold: %.3f", conf_score)
    return conf_score


def _compute_conf_thres(
    scores_full: list[list[float]],
    num_gts_per_class: np.ndarray,
    other_sign_class: int,
    iou_idx: int,
):
    num_classes = len(scores_full)
    num_ious = len(scores_full[0])
    scores_thres = np.linspace(0, 1, _NUM_SCORES)
    tp_score_full = np.zeros((_NUM_SCORES, num_ious, num_classes))
    fp_score_full = np.zeros_like(tp_score_full)

    # Get true and false positive at all values of score thres
    for i, thres in enumerate(scores_thres):
        tp_full, fp_full = _get_tp_fp_full(scores_full, thres)
        tp_score_full[i] = tp_full
        fp_score_full[i] = fp_full

    # Compute f1 scores
    recall = tp_score_full / (num_gts_per_class[None, None, :] + _EPS)
    precision = tp_score_full / (tp_score_full + fp_score_full + _EPS)
    f1_scores = 2 * precision * recall / (precision + recall + _EPS)
    assert np.all(f1_scores >= 0) and not np.any(np.isnan(f1_scores))

    if config_base["use_per_class_conf_thres"]:
        # Remove 'other' class from f1 and select desired IoU thres
        f1_scores[:, iou_idx, other_sign_class] -= 1e9
        f1_score = f1_scores[:, iou_idx]
        # Compute score thres per class
        class_idx = np.arange(num_classes)
        max_f1_idx = f1_score.argmax(0)
        max_f1 = f1_score[max_f1_idx, class_idx]
        tp = tp_score_full[max_f1_idx, iou_idx, class_idx]
        fp = fp_score_full[max_f1_idx, iou_idx, class_idx]
        conf_thres = np.take(scores_thres, max_f1_idx)

        def array2str(array):
            return np.array2string(array, separator=", ")

        logger.info("max_f1_idx: %s", array2str(max_f1_idx))
        logger.info("max_f1: %s", array2str(max_f1))
        logger.info("conf_thres: %s", array2str(conf_thres))
    else:
        # Remove 'other' class from f1 and select desired IoU thres
        f1_score = np.delete(f1_scores[:, iou_idx], other_sign_class, axis=1)
        # Compute one score thres for all class (use mean f1 score)
        f1_mean = f1_score.mean(2)
        max_f1_idx = f1_mean.argmax()
        max_f1 = f1_mean[max_f1_idx]
        tp = tp_score_full[iou_idx, :, max_f1_idx]
        fp = fp_score_full[iou_idx, :, max_f1_idx]
        conf_thres = scores_thres[max_f1_idx]
        logger.info("max_f1_idx: %d", max_f1_idx)
        logger.info("max_f1: %.4f", max_f1)
        logger.info("conf_thres: %.3f", conf_thres)
    return tp, fp, conf_thres


def _compute_metrics(
    scores_full: np.ndarray,
    num_gts_per_class: np.ndarray,
    conf_thres: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Compute true positives, false positives, and score threshold."""
    other_sign_class: int = config_base["other_sign_class"]
    iou_thres: float = config_base["iou_thres"]
    all_iou_thres = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    iou_idx = np.where(all_iou_thres == iou_thres)[0]
    # iou_idx can be [0], and this evaluate to True
    if len(iou_idx) == 0:
        raise ValueError(f"Invalid iou_thres {iou_thres}!")
    iou_idx = int(iou_idx)

    if config_base["compute_conf_thres"]:
        # Find score threshold that maximizes F1 score
        logger.info(
            "conf_thres not specified. Finding one that maximizes F1 scores..."
        )
        tp, fp, conf_thres = _compute_conf_thres(
            scores_full,
            num_gts_per_class,
            other_sign_class,
            iou_idx,
        )
    else:
        logger.info("Using specified conf_thres of %s...", str(conf_thres))
        tp_full, fp_full = _get_tp_fp_full(scores_full, conf_thres)
        tp, fp = tp_full[iou_idx], fp_full[iou_idx]

    recall = tp / (num_gts_per_class + _EPS)
    precision = tp / (tp + fp + _EPS)

    # Compute combined metrics, ignoring class
    recall_cmb = tp.sum() / (num_gts_per_class.sum() + _EPS)

    logger.info("num_gts_per_class: %s", str(num_gts_per_class))
    logger.info("tp: %s", str(tp))
    logger.info("fp: %s", str(fp))
    logger.info("precision: %s", str(precision))
    logger.info("recall: %s", str(recall))
    logger.info("recall_cmb: %s", str(recall_cmb))

    return tp, fp, conf_thres


def _dump_results(results: Dict[str, Any]) -> None:
    """Dump result dict to pickle file.

    Use hash of eval and attack configs for naming so only one result is saved
    per setting.

    Args:
        results: Result dict.
        config_base: Evaluation config dict.
    """
    result_dir = config_base["result_dir"]
    debug = config_base["debug"]
    if debug:
        return
    # Keep only eval params that matter (uniquely identifies evaluation setting)
    cfg_eval = {}
    for param in _EVAL_PARAMS:
        cfg_eval[param] = config_base[param]

    # Compute hash of both dicts to use as naming so we only keep one copy of
    # result in the exact same setting.
    config_base_hash = _hash_dict(cfg_eval)
    # Attack params are already contained in name
    config_attack_hash = _hash_dict({"name": config_base["name"]})
    result_path = os.path.join(
        result_dir,
        (
            f"results_eval{config_base_hash}_atk{config_attack_hash}_"
            f"split{config_base['split_file_hash']}.pkl"
        ),
    )
    with open(result_path, "wb") as file:
        pickle.dump(results, file)

    if config_base["compute_conf_thres"]:
        # Try to load existing metadata
        metadata_dir = os.path.join(
            config_base["base_dir"], config_base["name"], "metadata.pkl"
        )
        metadata = {}
        if os.path.isfile(metadata_dir):
            logger.info("Existing metadata file found at %s.", metadata_dir)
            with open(metadata_dir, "rb") as file:
                metadata = pickle.load(file)

        if "conf_thres" not in metadata:
            logger.info(
                "conf_thres does not exist in metadata. Creating an empty "
                "one..."
            )
            metadata["conf_thres"] = [
                None for _ in LABEL_LIST[config_base["dataset"]]
            ]

        # Write new conf_thres
        conf_thres = results["bbox"]["conf_thres"]
        if isinstance(conf_thres, list) or config_base["obj_class"] == -1:
            metadata["conf_thres"] = conf_thres
        else:
            metadata["conf_thres"][config_base["obj_class"]] = conf_thres

        with open(metadata_dir, "wb") as file:
            pickle.dump(metadata, file)
        logger.info("Metadata is saved at %s.", metadata_dir)

    logger.info("Results are saved at %s.", result_dir)


def main() -> None:
    """Main function."""
    attack_config_path: str = config_base["attack_config_path"]
    class_names: List[str] = LABEL_LIST[config_base["dataset"]]

    # Load adversarial patch and config
    if os.path.isfile(attack_config_path):
        logger.info(
            "Loading saved attack config from %s...", attack_config_path
        )
        with open(attack_config_path, "r", encoding="utf-8") as file:
            # pylint: disable=unexpected-keyword-arg
            config_attack = yaml.safe_load(file, Loader=yaml.FullLoader)
    else:
        config_attack = config["attack"]

    # Build model
    model = detectron2.engine.DefaultPredictor(cfg).model

    # Build dataloader
    dataloader, split_file_names = data_util.get_dataloader(config_base)
    # Keep hash of split files in config eval for naming dumped results
    config_base["split_file_hash"] = _hash(str(sorted(split_file_names)))

    evaluator = detectron_evaluator.DetectronEvaluator(
        config,
        model,
        dataloader,
        class_names=class_names,
    )
    logger.info("=> Running attack...")
    _, metrics = evaluator.run()

    eval_cfg = _normalize_dict(config_base)
    results: Dict[str, Any] = {**metrics, **eval_cfg, **config_attack}

    # Logging results
    metrics: Dict[str, Any] = results["bbox"]
    conf_thres: float = config_base["conf_thres"]

    if config_base["synthetic"]:
        obj_class: int = config["base"]["obj_class"]
        if obj_class is None or obj_class < 0:
            raise ValueError(f"Invalid obj_class value {obj_class}!")
        iou_thres_idx = int(
            np.where(metrics["all_iou_thres"] == config_base["iou_thres"])[0]
        )
        total_num_patches = metrics["total_num_patches"]
        syn_scores = metrics["syn_scores"]
        syn_matches = metrics["syn_matches"]

        if config_base["compute_conf_thres"]:
            # Compute conf_thres for synthetic data using desired FNR
            desired_fnr = config["base"]["syn_desired_fnr"]
            if isinstance(desired_fnr, (list, tuple)):
                desired_fnr = desired_fnr[obj_class]
            conf_thres = _compute_conf_thres_syn(
                (syn_scores * syn_matches)[iou_thres_idx],
                desired_fnr=desired_fnr,
            )
        if isinstance(conf_thres, (list, tuple)):
            conf_thres = conf_thres[obj_class]
        metrics["conf_thres"] = conf_thres

        # Get detection for desired score and for all IoU thresholds
        detected = (syn_scores >= conf_thres) * syn_matches
        # Select desired IoU threshold
        tp = detected[iou_thres_idx].sum()
        fn = total_num_patches - tp
        metrics["syn_total"] = total_num_patches
        metrics["syn_tp"] = int(tp)
        metrics["syn_fn"] = int(fn)
        metrics["syn_tpr"] = tp / total_num_patches
        metrics["syn_fnr"] = fn / total_num_patches
        logger.info(
            "[Syn] Total: %4d\nTP: %4d (%.4f)\nFN: %4d (%.4f)\n",
            metrics["syn_total"],
            metrics["syn_tp"],
            metrics["syn_tpr"],
            metrics["syn_fn"],
            metrics["syn_fnr"],
        )
    else:
        num_gts_per_class = metrics["num_gts_per_class"]
        tp, fp, conf_thres = _compute_metrics(
            metrics["scores_full"],
            num_gts_per_class,
            conf_thres,
        )
        metrics["conf_thres"] = conf_thres

        for key, value in metrics.items():
            if "syn" in key or not isinstance(value, (int, float, str, bool)):
                continue
            logger.info("%s: %s", key, str(value))

        logger.info("          tp   fp   num_gt")
        tp_all, fp_all, total = 0, 0, 0
        for i, (t, f, num_gt) in enumerate(zip(tp, fp, num_gts_per_class)):
            logger.info(
                "Class %2d: %4d %4d %4d", i, int(t), int(f), int(num_gt)
            )
            metrics[f"TP-{class_names[i]}"] = t
            metrics[f"FP-{class_names[i]}"] = f
            tp_all += t
            fp_all += f
            total += num_gt
        metrics["TPR-all"] = tp_all / total
        metrics["FPR-all"] = fp_all / total
        logger.info("Total num patches: %d", metrics["total_num_patches"])

    _dump_results(results)
    logger.info("Finished.")


if __name__ == "__main__":
    config: Dict[str, Dict[str, Any]] = reap_args_parser(
        True, is_gen_patch=False
    )
    cfg = setup_detectron_cfg(config)

    config_base: Dict[str, Any] = config["base"]
    seed: int = config_base["seed"]

    # Set up logger to both stdout and log file
    log_level: int = (
        logging.DEBUG
        if config_base["debug"] or config_base["verbose"]
        else logging.INFO
    )
    FORMAT_STR = "[%(asctime)s - %(name)s - %(levelname)s]: %(message)s"
    formatter = logging.Formatter(FORMAT_STR)
    logging.basicConfig(
        stream=sys.stdout,
        format=FORMAT_STR,
        level=log_level,
    )
    file_handler = logging.FileHandler(
        os.path.join(config_base["result_dir"], "results.log"), mode="a"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(config)

    dt_log = logging.getLogger("detectron2")
    dt_log.setLevel(log_level)
    dt_log = logging.getLogger("fvcore")
    dt_log.setLevel(log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Set random seeds
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Register Detectron2 dataset
    data_util.register_dataset(config_base)

    main()
