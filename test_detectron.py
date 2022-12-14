"""Test script for Detectron2 models."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import detectron2
import numpy as np
import pandas as pd
import torch
import yaml
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO

import adv_patch_bench.dataloaders.detectron.util as data_util
from adv_patch_bench.dataloaders.detectron import custom_build, mapper
from adv_patch_bench.evaluators import detectron_evaluator
from adv_patch_bench.utils.argparse import (
    eval_args_parser,
    setup_detectron_test_args,
)
from adv_patch_bench.utils.types import DetectronSample
from hparams import LABEL_LIST

log = logging.getLogger(__name__)
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


def _get_img_ids(dataset: str, obj_class: int) -> List[int]:
    """Get ids of images that contain desired object class."""
    metadata = detectron2.data.MetadataCatalog.get(dataset)
    if not hasattr(metadata, "json_file"):
        cache_path = os.path.join(cfg.OUTPUT_DIR, f"{dataset}_coco_format.json")
        metadata.json_file = cache_path
        convert_to_coco_json(dataset, cache_path)

    json_file = PathManager.get_local_path(metadata.json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    img_ids: List[int] = coco_api.getImgIds(catIds=[obj_class])
    return img_ids


def _get_filename_from_id(
    data_dicts: List[DetectronSample], img_ids: List[int]
) -> List[str]:
    filenames: List[str] = []
    img_ids_set = set(img_ids)
    for data in data_dicts:
        if data["image_id"] in img_ids_set:
            filenames.append(data["file_name"].split("/")[-1])
    return filenames


def _hash_dict(config_dict: Dict[str, Any]) -> str:
    dict_str = json.dumps(config_dict, sort_keys=True)
    # Take first 8 characters of the hash since we prefer short file name
    return _hash(dict_str)


def _normalize_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
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
        log.debug(
            f"max_f1_idx: {max_f1_idx}, max_f1: {max_f1:.4f}, conf_thres: "
            f"{conf_thres:.3f}."
        )

    else:

        log.debug("Using specified conf_thres of %f...", conf_thres)

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

    log.debug(f"num_gts_per_class: {num_gts_per_class}")
    log.debug(f"tp: {tp}")
    log.debug(f"fp: {fp}")
    log.debug(f"precision: {pr}")
    log.debug(f"recall: {rc}")
    log.debug(f"recall_cmb: {recall_cmb}")

    return tp, fp, conf_thres


def _dump_results(
    results: Dict[str, Any],
    config_eval: Dict[str, Any],
) -> None:
    """Dump result dict to pickle file.

    Use hash of eval and attack configs for naming so only one result is saved
    per setting.

    Args:
        results: Result dict.
        config_eval: Evaluation config dict.
    """
    result_dir = config_eval["result_dir"]
    debug = config_eval["debug"]
    if debug:
        return
    # Keep only eval params that matter (uniquely identifies evaluation setting)
    cfg_eval = {}
    for param in _EVAL_PARAMS:
        cfg_eval[param] = config_eval[param]

    # Compute hash of both dicts to use as naming so we only keep one copy of
    # result in the exact same setting.
    config_eval_hash = _hash_dict(cfg_eval)
    # Attack params are already contained in name
    config_attack_hash = _hash_dict({"name": config_eval["name"]})
    result_path = os.path.join(
        result_dir,
        (
            f"results_eval{config_eval_hash}_atk{config_attack_hash}_"
            f"split{config_eval['split_file_hash']}.pkl"
        ),
    )
    with open(result_path, "wb") as f:
        pickle.dump(results, f)


def main(config: Dict[str, Dict[str, Any]]):
    """Main function.

    Args:
        config: Config dict for both eval and attack.
    """
    config_eval: Dict[str, Any] = config["eval"]
    dataset: str = config_eval["dataset"]
    attack_config_path: str = config_eval["attack_config_path"]
    split_file_path: str = config_eval["split_file_path"]
    class_names: List[str] = LABEL_LIST[dataset]

    # Load adversarial patch and config
    if os.path.isfile(attack_config_path):
        log.info(f"Loading saved attack config from {attack_config_path}...")
        with open(attack_config_path) as f:
            # pylint: disable=unexpected-keyword-arg
            config_attack = yaml.safe_load(f, Loader=yaml.FullLoader)
    else:
        config_attack = config["attack"]

    # Build model
    model = detectron2.engine.DefaultPredictor(cfg).model

    # Build dataloader
    # First, get list of file names to evaluate on
    data_dicts: List[DetectronSample] = detectron2.data.DatasetCatalog.get(
        config_eval["dataset"]
    )
    split_file_names: Optional[List[str]] = None
    if split_file_path is not None:
        print(f"Loading file names from {split_file_path}...")
        with open(split_file_path, "r") as f:
            split_file_names = set(f.read().splitlines())

    # Filter only images with desired class when evaluating on REAP
    if dataset == "reap":
        img_ids = _get_img_ids(dataset, config_eval["obj_class"])
        class_file_names = set(_get_filename_from_id(data_dicts, img_ids))
        split_file_names = split_file_names.intersection(class_file_names)

    # Keep hash of split files in config eval for naming dumped results
    config_eval["split_file_hash"] = _hash(str(sorted(split_file_names)))

    dataloader = custom_build.build_detection_test_loader(
        data_dicts,
        mapper=mapper.BenignMapper(cfg, is_train=False),
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        split_file_names=split_file_names,
    )

    evaluator = detectron_evaluator.DetectronEvaluator(
        config_eval,
        config_attack,
        model,
        dataloader,
        class_names=class_names,
    )
    log.info("=> Running attack...")
    _, metrics = evaluator.run()

    eval_cfg = _normalize_dict(config_eval)
    results: Dict[str, Any] = {**metrics, **eval_cfg, **config_attack}
    _dump_results(results, config_eval)

    # Logging results
    metrics: Dict[str, Any] = results["bbox"]
    conf_thres: float = config_eval["conf_thres"]

    if config_eval["synthetic"]:
        total_num_patches = metrics["total_num_patches"]
        syn_scores = metrics["syn_scores"]
        syn_matches = metrics["syn_matches"]
        all_iou_thres = metrics["all_iou_thres"]
        iou_thres = config_eval["iou_thres"]

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
        log.info(
            f'[Syn] Total: {metrics["syn_total"]:4d}\n'
            f'      TP: {metrics["syn_tp"]:4d} ({metrics["syn_tpr"]:.4f})\n'
            f'      FN: {metrics["syn_fn"]:4d} ({metrics["syn_fnr"]:.4f})\n'
        )
    else:
        num_gts_per_class = metrics["num_gts_per_class"]
        tp, fp, conf_thres = _compute_metrics(
            metrics["scores_full"],
            num_gts_per_class,
            config_eval["other_sign_class"],
            conf_thres,
            config_eval["iou_thres"],
        )
        if config_eval["conf_thres"] is None:
            # Update with new conf_thres
            metrics["conf_thres"] = conf_thres

        for k, v in metrics.items():
            if "syn" in k or not isinstance(v, (int, float, str, bool)):
                continue
            log.info(f"{k}: {v}")

        log.info("          tp   fp   num_gt")
        tp_all = 0
        fp_all = 0
        total = 0
        for i, (t, f, n) in enumerate(zip(tp, fp, num_gts_per_class)):
            log.info(f"Class {i:2d}: {int(t):4d} {int(f):4d} {int(n):4d}")
            metrics[f"TP-{class_names[i]}"] = t
            metrics[f"FP-{class_names[i]}"] = f
            tp_all += t
            fp_all += f
            total += n
        metrics["TPR-all"] = tp_all / total
        metrics["FPR-all"] = fp_all / total
        log.info(f'Total num patches: {metrics["total_num_patches"]}')
        _dump_results(results, config_eval)


if __name__ == "__main__":
    config: Dict[str, Dict[str, Any]] = eval_args_parser(
        True, is_gen_patch=False
    )
    cfg = setup_detectron_test_args(config)

    config_eval: Dict[str, Any] = config["eval"]
    seed: int = config_eval["seed"]
    log_level: int = logging.DEBUG if config_eval["debug"] else logging.WARNING

    # Set up logger
    log.setLevel(log_level)
    file_handler = logging.FileHandler(
        os.path.join(config_eval["result_dir"], "results.log"), mode="a"
    )
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info(config)

    dt_log = logging.getLogger("detectron2")
    dt_log.setLevel(log_level)
    dt_log = logging.getLogger("fvcore")
    dt_log.setLevel(log_level)

    # Set random seeds
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Register Detectron2 dataset
    data_util.register_dataset(config_eval)

    main(config)
