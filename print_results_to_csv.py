"""Print metrics in readable format given raw results."""

import argparse
import pathlib
import pickle
from typing import List

import numpy as np
import pandas as pd

from hparams import LABEL_LIST

_DATASET = "reap"
_NUM_CLASSES = len(LABEL_LIST[_DATASET]) - 1
_NUM_SIGNS_PER_CLASS = np.zeros(_NUM_CLASSES, dtype=np.int64)
_NUM_IOU_THRES = 10
BASE_PATH = "./results/"
# CONF_THRES = 0.634  # FIXME
# CONF_THRES_1 = [0.949,0.950,0.898,0.906,0.769,0.959,0.732,0.538,0.837,0.862,0.823,0.0]
CONF_THRES_2 = [0.949,0.950,0.898,0.906,0.769,0.959,0.732,0.538,0.837,0.862,0.823,0.0]
SYN_CONF_THRES_2 = []
CONF_THRES_3 = [0.871,0.878,0.991,0.619,0.929,0.770,0.976,0.688,0.943,0.715,0.964,0.0]
SYN_CONF_THRES_3 = [0.907,0.990,0.994,0.358,0.975,0.948,0.998,0.980,0.984,0.988,0.972,0.0]
CONF_THRES_4 = [0.896,0.944,0.957,0.901,0.796,0.870,0.743,0.687,0.929,0.991,0.981,0.0]
SYN_CONF_THRES_4 = [0.874,0.959,0.961,0.369,0.636,0.993,0.993,0.995,0.893,0.999,0.955,0.0]
iou_idx = 0  # 0.5

_TRANSFORM_PARAMS: List[str] = [
    "interp",
    "reap_geo_method",
    "reap_relight_method",
    "reap_relight_polynomial_degree",
    "reap_relight_percentile",
    "syn_obj_width_px",
    "syn_rotate",
    "syn_scale",
    "syn_translate",
    "syn_colorjitter",
    "syn_3d_dist",
]


def _compute_ap_recall(
    scores, matched, NP, conf_thres=None, recall_thresholds=None
):
    """Compute AP, precision, and recall.

    This curve tracing method has some quirks that do not appear
    when only unique confidence thresholds are used (i.e. Scikit-learn's
    implementation), however, in order to be consistent, the COCO's method is
    reproduced.
    """
    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    score_idx = None
    if conf_thres is not None:
        score_idx = np.where(scores >= conf_thres)[0][-1]

    return {
        "precision": pr[score_idx],
        "recall": rc[score_idx],
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }


def _average(print_df_rows, base_sid, all_class_sid, metric_name):
    metrics = np.zeros(len(_NUM_SIGNS_PER_CLASS))
    for i in range(len(_NUM_SIGNS_PER_CLASS)):
        sid = f"{base_sid} | {i:02d}"
        if sid not in print_df_rows:
            continue
        metrics[i] = print_df_rows[f"{base_sid} | {i:02d}"][metric_name]
    print_df_rows[all_class_sid][metric_name] = np.mean(metrics)
    return metrics


def main(args):
    """Main function."""
    exp_type = args.exp_type
    clean_exp_name = args.clean_exp_name
    attack_exp_name = args.attack_exp_name
    clean_exp_path = pathlib.Path(BASE_PATH) / clean_exp_name
    attack_exp_path = pathlib.Path(BASE_PATH) / attack_exp_name
    exp_paths = []
    if clean_exp_path.is_dir():
        exp_paths.extend(list(clean_exp_path.iterdir()))
    if attack_exp_path.is_dir():
        exp_paths.extend(list(attack_exp_path.iterdir()))

    model_id = int(clean_exp_name.split("model")[1])
    is_syn = "synthetic" in clean_exp_name
    conf_id = f'{model_id}-{is_syn}'
    conf_thres = {
        "2-False": CONF_THRES_2,
        "2-True": SYN_CONF_THRES_2,
        "3-False": CONF_THRES_3,
        "3-True": SYN_CONF_THRES_3,
        "4-False": CONF_THRES_4,
        "4-True": SYN_CONF_THRES_4,
    }[conf_id]

    df_rows = {}
    gt_scores = [{}, {}]
    results_all_classes = {}
    print_df_rows = {}
    tp_scores = {}
    fp_scores = {}
    repeated_results = []

    # Iterate over sign classes
    for sign_path in exp_paths:

        if not sign_path.is_dir():
            continue

        # Iterate over attack_type (none, load, syn_none, syn_load, etc.)
        for setting_path in sign_path.iterdir():
            result_paths = setting_path.glob("*.pkl")
            result_paths = list(result_paths)
            if not result_paths:
                continue

            # Select latest result only
            # mtimes = np.array(
            #     [
            #         float(pathlib.Path(result_path).stat().st_mtime)
            #         for result_path in result_paths
            #     ]
            # )
            # latest_idx = np.argmax(mtimes)
            # result_paths = [result_paths[latest_idx]]

            # Iterate over result pickle files
            for result_path in result_paths:

                result_path = str(result_path)
                with open(result_path, "rb") as f:
                    results = pickle.load(f)

                if "obj_class" not in results:
                    continue

                # Add timestamp
                # time = result_path.split("_")[-1].split(".pkl")[0]
                result_name = result_path.split("/")[-1]
                # obj_class_name = result_path.split("/")[-3]
                hashes = result_name.split("_")[1:]
                eval_hash = hashes[0].split("eval")[1]
                eval_hash = "dummy"
                # atk_hash = hashes[1].split("atk")[1]
                # if len(hashes) < 3:
                #     split_hash = "null"
                # else:
                #     split_hash = hashes[2].split("split")[1].split(".pkl")[0]
                if "bbox" not in results:
                    continue
                metrics = results["bbox"]
                print(result_path)

                # Experiment setting identifier for matching clean and attack
                obj_class = results["obj_class"]
                if obj_class < 0:
                    continue
                synthetic = int(results["synthetic"])
                attack_type = results["attack_type"]
                is_attack = int(results["attack_type"] != "none")
                scores_dict = gt_scores[is_attack]

                if synthetic:
                    # Synthetic sign
                    if exp_type is not None and exp_type != "syn":
                        continue
                    cls_scores = {
                        obj_class: metrics["syn_scores"]
                        * metrics["syn_matches"]
                    }
                    token_list = []
                    for param in _TRANSFORM_PARAMS:
                        if "syn" in param:
                            token_list.append(str(results[param]))
                    base_sid = f"syn | {attack_type} | " + "_".join(token_list)
                    # base_sid += "_atk1" if is_attack else "_atk0"
                else:
                    # Real signs
                    if exp_type is not None and exp_type != "reap":
                        continue
                    if "gtScores" not in metrics:
                        continue
                    cls_scores = metrics["gtScores"]
                    tf_mode = results.get("reap_geo_method", "perspective")
                    rl_mode = results["reap_relight_method"]
                    base_sid = f"reap | {attack_type} | {tf_mode} | {rl_mode}"
                base_sid += f" | {eval_hash}"

                if base_sid not in tp_scores:
                    tp_scores[base_sid] = {t: [] for t in range(10)}
                    fp_scores[base_sid] = {t: [] for t in range(10)}

                scores = cls_scores[obj_class]
                num_gts = scores.shape[1]
                _NUM_SIGNS_PER_CLASS[obj_class] = num_gts
                sid = f"{base_sid} | {obj_class:02d}"
                if sid in scores_dict:
                    repeated_results.append(result_path)
                    continue
                scores_dict[sid] = scores

                tp = np.sum(scores[iou_idx] >= conf_thres[obj_class])
                class_name = LABEL_LIST[_DATASET][obj_class]
                tpr = tp / num_gts
                metrics[f"FNR-{class_name}"] = 1 - tpr

                print_df_rows[sid] = {
                    "id": sid,
                    "eval_hash": eval_hash,
                    "attack_type": attack_type,
                    "FNR": (1 - tpr) * 100,
                }
                if not synthetic:
                    # Collect AP, precision, and recall
                    scores_full = results["bbox"]["scores_full"][obj_class]
                    scores_tp = scores_full[iou_idx][0]
                    scores_fp = scores_full[iou_idx][1]
                    scores = np.concatenate([scores_tp, scores_fp], axis=0)
                    matches = np.zeros_like(scores, dtype=bool)
                    num_matched = len(scores_tp)
                    matches[:num_matched] = 1
                    outputs = _compute_ap_recall(
                        scores,
                        matches,
                        num_gts,
                        conf_thres=conf_thres[obj_class],
                    )
                    # FIXME: precision can't be weighted average
                    print_df_rows[sid]["Precision"] = outputs["precision"] * 100
                    print_df_rows[sid]["Recall"] = outputs["recall"] * 100
                    print_df_rows[sid]["AP"] = results["bbox"]["AP"]
                    for t in range(10):
                        tp_scores[base_sid][t].extend(scores_full[t][0])
                        fp_scores[base_sid][t].extend(scores_full[t][1])

                # Create DF row for all classes
                all_class_sid = f"{base_sid} | all"
                print_df_rows[all_class_sid] = {
                    "id": all_class_sid,
                    "eval_hash": eval_hash,
                    "attack_type": attack_type,
                }
                # Weighted
                allw_class_sid = f"{base_sid} | allw"
                print_df_rows[allw_class_sid] = {
                    "id": allw_class_sid,
                    "eval_hash": eval_hash,
                    "attack_type": attack_type,
                }

                # Print result as one row in df
                df_row = {}
                for k, v in results.items():
                    if isinstance(v, (float, int, str, bool)):
                        df_row[k] = v
                for k, v in metrics.items():
                    if isinstance(v, (float, int, str, bool)):
                        df_row[k] = v
                df_rows[sid] = df_row

    # FNR for clean syn
    fnrs = np.zeros(_NUM_CLASSES)
    sid_no_class = None
    for sid, data in print_df_rows.items():
        if data["attack_type"] != "none":
            continue
        base_sid = " | ".join(sid.split(" | ")[:-1])
        all_class_sid = f"{base_sid} | all"
        allw_class_sid = f"{base_sid} | allw"

        if "reap" in sid:
            _average(print_df_rows, base_sid, all_class_sid, "Precision")
            _average(print_df_rows, base_sid, all_class_sid, "Recall")
            _average(print_df_rows, base_sid, all_class_sid, "AP")
        fnrs = _average(print_df_rows, base_sid, all_class_sid, "FNR")
        print_df_rows[allw_class_sid]["FNR"] = np.sum(
            fnrs * _NUM_SIGNS_PER_CLASS / np.sum(_NUM_SIGNS_PER_CLASS)
        )

    # Iterate through all attack experiments
    for sid, adv_scores in gt_scores[1].items():

        split_sid = sid.split(" | ")
        k = int(split_sid[-1])
        # Find results without attack in the same setting
        clean_sid = " | ".join([split_sid[0], "none", *split_sid[2:]])
        if clean_sid not in gt_scores[0]:
            continue

        clean_scores = gt_scores[0][clean_sid]
        clean_detected = clean_scores[iou_idx] >= conf_thres[k]
        adv_detected = adv_scores[iou_idx] >= conf_thres[k]
        total = clean_scores.shape[1]

        num_succeed = np.sum(~adv_detected & clean_detected)
        num_clean = np.sum(clean_detected)

        attack_success_rate = num_succeed / (num_clean + 1e-9) * 100
        df_rows[sid]["ASR"] = attack_success_rate
        print_df_rows[sid]["ASR"] = attack_success_rate

        sid_no_class = " | ".join(split_sid[:-1])
        fnr = print_df_rows[sid]["FNR"]
        if "reap" in sid_no_class:
            ap = print_df_rows[sid]["AP"]
        else:
            ap = -1e9

        if sid_no_class in results_all_classes:
            results_all_classes[sid_no_class]["num_succeed"] += num_succeed
            results_all_classes[sid_no_class]["num_clean"][k] = num_clean
            results_all_classes[sid_no_class]["num_total"] += total
            results_all_classes[sid_no_class]["asr"][k] = attack_success_rate
            results_all_classes[sid_no_class]["fnr"][k] = fnr
            results_all_classes[sid_no_class]["ap"][k] = ap
        else:
            asrs = np.zeros(_NUM_CLASSES)
            asrs[k] = attack_success_rate
            fnrs = np.zeros_like(asrs)
            fnrs[k] = fnr
            aps = np.zeros_like(asrs)
            aps[k] = ap
            num_cleans = np.zeros_like(asrs)
            num_cleans[k] = num_clean
            results_all_classes[sid_no_class] = {
                "num_succeed": num_succeed,
                "num_clean": num_cleans,
                "num_total": total,
                "asr": asrs,
                "fnr": fnrs,
                "ap": aps,
            }

    df_rows = list(df_rows.values())
    df = pd.DataFrame.from_records(df_rows)
    df = df.sort_index(axis=1)
    # df.to_csv(attack_exp_path / "results.csv")

    print(attack_exp_name, clean_exp_name, conf_thres)
    print("All-class ASR")
    for sid, result in results_all_classes.items():

        num_succeed = result["num_succeed"]
        num_clean = result["num_clean"]
        total = result["num_total"]
        asr = num_succeed / (num_clean.sum() + 1e-9) * 100

        # Average metrics over classes instead of counting all as one
        all_class_sid = f"{sid} | all"
        asrs = result["asr"]
        fnrs = result["fnr"]
        avg_asr = np.mean(asrs)
        print_df_rows[all_class_sid]["ASR"] = avg_asr
        avg_fnr = np.mean(fnrs)
        print_df_rows[all_class_sid]["FNR"] = avg_fnr

        # Weighted average by number of real sign distribution
        allw_class_sid = f"{sid} | allw"
        print_df_rows[allw_class_sid]["ASR"] = np.sum(
            asrs * _NUM_SIGNS_PER_CLASS / np.sum(_NUM_SIGNS_PER_CLASS)
        )
        print_df_rows[allw_class_sid]["FNR"] = np.sum(
            fnrs * _NUM_SIGNS_PER_CLASS / np.sum(_NUM_SIGNS_PER_CLASS)
        )

        if "reap" in sid:
            # This is the correct (or commonly used) definition of mAP
            mAP = np.mean(result["ap"])
            print_df_rows[all_class_sid]["AP"] = mAP

            aps = np.zeros(_NUM_IOU_THRES)
            num_dts = None
            for t in range(_NUM_IOU_THRES):
                matched_len = len(tp_scores[sid][t])
                unmatched_len = len(fp_scores[sid][t])
                if num_dts is not None:
                    assert num_dts == matched_len + unmatched_len
                num_dts = matched_len + unmatched_len
                scores = np.zeros(num_dts)
                matches = np.zeros_like(scores, dtype=bool)
                scores[:matched_len] = tp_scores[sid][t]
                scores[matched_len:] = fp_scores[sid][t]
                matches[:matched_len] = 1
                aps[t] = _compute_ap_recall(scores, matches, total)["AP"]
            print_df_rows[allw_class_sid]["AP"] = np.mean(aps) * 100

        print(
            f"{sid}: combined {asr:.2f} ({num_succeed}/{num_clean.sum()}), "
            f"average {avg_asr:.2f}, total {total}"
        )

    for sid, tp_score in tp_scores.items():
        if "reap" in sid and "none" in sid:
            aps = np.zeros(_NUM_IOU_THRES)
            num_dts = None
            for t in range(_NUM_IOU_THRES):
                matched_len = len(tp_score[t])
                unmatched_len = len(fp_scores[sid][t])
                if num_dts is not None:
                    assert num_dts == matched_len + unmatched_len
                num_dts = matched_len + unmatched_len
                scores = np.zeros(num_dts)
                matches = np.zeros_like(scores, dtype=bool)
                scores[:matched_len] = tp_score[t]
                scores[matched_len:] = fp_scores[sid][t]
                matches[:matched_len] = 1
                aps[t] = _compute_ap_recall(
                    scores, matches, _NUM_SIGNS_PER_CLASS.sum()
                )["AP"]
            print_df_rows[sid + " | allw"]["AP"] = np.mean(aps) * 100

    print_df_rows = list(print_df_rows.values())
    df = pd.DataFrame.from_records(print_df_rows)
    df = df.sort_values(["id", "attack_type"])
    df = df.drop(columns=["attack_type"])
    # df = df.reindex(columns=["id", "FNR", "ASR", "AP", "Precision", "Recall"])
    df = df.reindex(columns=["id", "FNR", "ASR", "AP"])
    print(df.to_csv(float_format="%0.2f", index=False))
    print("Repeated results:", repeated_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_exp_name", type=str, help="clean_exp_name")
    parser.add_argument("attack_exp_name", type=str, help="attack_exp_name")
    parser.add_argument(
        "--exp_type",
        type=str,
        default=None,
        required=False,
        help="reap or syn (default is both)",
    )
    args = parser.parse_args()
    main(args)
