"""Test script for Detectron2 models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

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
import numpy as np
import pandas as pd
import torch
import yaml

import adv_patch_bench.dataloaders.detectron.util as data_util
from adv_patch_bench.evaluators import detectron_evaluator
from adv_patch_bench.utils.argparse import reap_args_parser, setup_detectron_cfg
from hparams import LABEL_LIST

logger = logging.getLogger(__name__)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")

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


def _compute_metrics(
    scores_full: np.ndarray,
    num_gts_per_class: np.ndarray,
    other_sign_class: int,
    conf_thres: Optional[float] = None,
    iou_thres: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:

    all_iou_thres = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    iou_idx = np.where(all_iou_thres == iou_thres)[0]
    # iou_idx can be [0], and this evaluate to True
    if len(iou_idx) == 0:
        raise ValueError(f"Invalid iou_thres {iou_thres}!")
    iou_idx = int(iou_idx)

    # Find score threshold that maximizes F1 score
    eps = np.spacing(1)
    num_classes = len(scores_full)
    num_ious = len(scores_full[0])

    if conf_thres is None:
        num_scores = 1000
        scores_thres = np.linspace(0, 1, num_scores)
        tp_full = np.zeros((num_ious, num_classes, num_scores))
        fp_full = np.zeros_like(tp_full)

        for t in range(num_ious):
            for k in range(num_classes):
                for si, s in enumerate(scores_thres):
                    tp_full[t, k, si] = np.sum(
                        np.array(scores_full[k][t][0]) >= s
                    )
                    fp_full[t, k, si] = np.sum(
                        np.array(scores_full[k][t][1]) >= s
                    )

        rc = tp_full / (num_gts_per_class[None, :, None] + eps)
        pr = tp_full / (tp_full + fp_full + eps)
        f1 = 2 * pr * rc / (pr + rc + eps)
        assert np.all(f1 >= 0) and not np.any(np.isnan(f1))

        # Remove 'other' class from f1 and average over remaining classes
        f1_mean = np.delete(f1[iou_idx], other_sign_class, axis=0).mean(0)
        max_f1_idx = f1_mean.argmax()
        max_f1 = f1_mean[max_f1_idx]
        tp: np.ndarray = tp_full[iou_idx, :, max_f1_idx]
        fp: np.ndarray = fp_full[iou_idx, :, max_f1_idx]
        conf_thres = scores_thres[max_f1_idx]
        logger.debug(
            "max_f1_idx: %d, max_f1: %.4f, conf_thres: %.3f.",
            max_f1_idx,
            max_f1,
            conf_thres,
        )
    else:
        logger.debug("Using specified conf_thres of %f...", conf_thres)

        tp_full = np.zeros((num_ious, num_classes))
        fp_full = np.zeros_like(tp_full)

        for t in range(num_ious):
            for k in range(num_classes):
                tp_full[t, k] = np.sum(
                    np.array(scores_full[k][t][0]) >= conf_thres
                )
                fp_full[t, k] = np.sum(
                    np.array(scores_full[k][t][1]) >= conf_thres
                )
        tp: np.ndarray = tp_full[iou_idx]
        fp: np.ndarray = fp_full[iou_idx]

    rc = tp / (num_gts_per_class + eps)
    pr = tp / (tp + fp + eps)

    # Compute combined metrics, ignoring class
    recall_cmb = tp.sum() / (num_gts_per_class.sum() + eps)

    logger.debug("num_gts_per_class: %s", str(num_gts_per_class))
    logger.debug("tp: %s", str(tp))
    logger.debug("fp: %s", str(fp))
    logger.debug("precision: %s", str(pr))
    logger.debug("recall: %s", str(rc))
    logger.debug("recall_cmb: %s", str(recall_cmb))

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
    logger.info("Results are saved at %s", result_dir)


def main() -> None:
    """Main function."""
    attack_config_path: str = config_base["attack_config_path"]
    class_names: List[str] = LABEL_LIST[config_base["dataset"]]

    # Load adversarial patch and config
    if os.path.isfile(attack_config_path):
        logger.info("Loading saved attack config from %s...", attack_config_path)
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
    _dump_results(results)

    # Logging results
    metrics: Dict[str, Any] = results["bbox"]
    conf_thres: float = config_base["conf_thres"]

    if config_base["synthetic"]:
        total_num_patches = metrics["total_num_patches"]
        syn_scores = metrics["syn_scores"]
        syn_matches = metrics["syn_matches"]
        all_iou_thres = metrics["all_iou_thres"]
        iou_thres = config_base["iou_thres"]

        # Get detection for desired score and for all IoU thresholds
        detected = (syn_scores >= conf_thres) * syn_matches
        # Select desired IoU threshold
        iou_thres_idx = int(np.where(all_iou_thres == iou_thres)[0])
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
            config_base["other_sign_class"],
            conf_thres,
            config_base["iou_thres"],
        )
        if config_base["conf_thres"] is None:
            # Update with new conf_thres
            metrics["conf_thres"] = conf_thres

        for key, value in metrics.items():
            if "syn" in key or not isinstance(value, (int, float, str, bool)):
                continue
            logger.info("%s: %s", key, str(value))

        logger.info("          tp   fp   num_gt")
        tp_all = 0
        fp_all = 0
        total = 0
        for i, (t, f, num_gt) in enumerate(zip(tp, fp, num_gts_per_class)):
            logger.info("Class %2d: %4d %4d %4d", i, int(t), int(f), int(num_gt))
            metrics[f"TP-{class_names[i]}"] = t
            metrics[f"FP-{class_names[i]}"] = f
            tp_all += t
            fp_all += f
            total += num_gt
        metrics["TPR-all"] = tp_all / total
        metrics["FPR-all"] = fp_all / total
        logger.info("Total num patches: %d", metrics["total_num_patches"])
        _dump_results(results)


if __name__ == "__main__":
    config: Dict[str, Dict[str, Any]] = reap_args_parser(
        True, is_gen_patch=False
    )
    cfg = setup_detectron_cfg(config)

    config_base: Dict[str, Any] = config["base"]
    seed: int = config_base["seed"]
    log_level: int = logging.DEBUG if config_base["debug"] else logging.WARNING

    # Set up logger
    logger.setLevel(log_level)
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

    # Set random seeds
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Register Detectron2 dataset
    data_util.register_dataset(config_base)

    main()
