# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import json
import os
import pdb
import pickle
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from adv_patch_bench.utils.argparse import (
    eval_args_parser,
    setup_yolo_test_args,
)

OTHER_SIGN_CLASS = 12
from hparams import SAVE_DIR_YOLO
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (
    LOGGER,
    box_iou,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    xywh2xyxy,
    xyxy2xywh,
)
# from yolov5.utils.metrics import ConfusionMatrix, ap_per_class_custom
# from yolov5.utils.plots import (
#     output_to_target,
#     plot_false_positives,
#     plot_images,
#     plot_val_study,
# )
from yolov5.utils.torch_utils import select_device, time_sync

warnings.filterwarnings("ignore")


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (
            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        )  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(
    detections,
    labels,
    iouv,
    other_class_label=None,
    other_class_confidence_threshold=0,
    match_on_iou_only=False,
):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(
        detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
    )
    iou = box_iou(labels[:, 1:5], detections[:, :4])

    if match_on_iou_only:
        x = torch.where((iou >= iouv[0]))  # IoU above threshold
    elif other_class_label:
        x = torch.where(
            (iou >= iouv[0])
            & (
                (labels[:, 0:1] == detections[:, 5])
                | (
                    (labels[:, 0:1] == other_class_label)
                    & (detections[:, 4] > other_class_confidence_threshold)
                )
            )
        )  # IoU above threshold and classes match
        # x = torch.where((iou >= iouv[0]))  # IoU above threshold
    else:
        # IoU above threshold and classes match
        x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    # else:
    #     x = torch.where((iou >= iouv[0]))  # IoU above threshold

    matches = []
    if x[0].shape[0]:
        # [label_idx, detection, iou]
        matches = (
            torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
            .cpu()
            .numpy()
        )
        if x[0].shape[0] > 1:
            # sort matches by decreasing order of IOU
            matches = matches[matches[:, 2].argsort()[::-1]]
            # for each (label, detection) pair, select the one with highest IOU score
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct, matches, iou


def populate_default_metric(lbl_, min_area, filename, other_class_label):
    lbl_ = lbl_.cpu().numpy()
    class_label, _, _, bbox_width, bbox_height, obj_id = lbl_
    bbox_area = bbox_width * bbox_height
    current_label_metric = {}
    current_label_metric["filename"] = filename
    current_label_metric["object_id"] = obj_id
    current_label_metric["label"] = class_label
    current_label_metric["correct_prediction"] = 0
    current_label_metric["prediction"] = None
    current_label_metric["sign_width"] = bbox_width
    current_label_metric["sign_height"] = bbox_height
    current_label_metric["confidence"] = None
    current_label_metric["too_small"] = bbox_area < min_area
    current_label_metric["iou"] = None
    # current_label_metric['too_small'] = bbox_area < min_area or class_label == other_class_label
    current_label_metric["changed_from_other_label"] = 0
    return current_label_metric


@torch.no_grad()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    **kwargs,
):

    weights = '/data/shared/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt'
    # Initialize/load model and set device
    LOGGER.info("Loading Model...")
    device = select_device(device, batch_size=batch_size)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, pt, jit, engine = (
        model.stride,
        model.pt,
        model.jit,
        model.engine,
    )
    # check image size
    imgsz = check_img_size(imgsz, s=stride)
    # half precision only supported by PyTorch on CUDA
    half &= (pt or jit or engine) and device.type != "cpu"
    if pt or jit:
        model.model.half() if half else model.model.float()
    elif engine:
        batch_size = model.batch_size
    else:
        half = False
        batch_size = 1  # export.py models default to batch-size 1
        device = torch.device("cpu")
        LOGGER.info(
            f"Forcing --batch-size 1 square inference shape"
            f"(1,3,{imgsz},{imgsz}) for non-PyTorch backends"
        )

    # Data
    data = check_dataset(data)  # check
    # Dataloader
    pad = 0.5
   
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=pt,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    s = ("%20s" + "%11s" * 6) % (
        "Class",
        "Images",
        "Labels",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )

    pbar = tqdm(
        dataloader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )  # progress bar

    shapes_df = pd.DataFrame()
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        for image_i, path in enumerate(paths):   
            
            (h0, w0), ((h_ratio, w_ratio), (w_pad, h_pad)) = shapes[image_i]
            # print(path)   
            # print(h0)

            filename = path.split("/")[-1]
            row = {'filename': filename, 'h0': h0, 'w0': w0, 'h_ratio': h_ratio, 'w_ratio': w_ratio, 'w_pad': w_pad, 'h_pad': h_pad}
            shapes_df = shapes_df.append(row, ignore_index=True)
            
    shapes_df.to_csv('shapes_df.csv')


def parse_opt():
    opt = eval_args_parser(False, root=ROOT)
    setup_yolo_test_args(opt, OTHER_SIGN_CLASS)
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)

    assert not (
        opt.synthetic and opt.attack_type == "per-sign"
    ), "Synthetic evaluation with per-sign attack is not implemented."

    return opt


# def main(opt):
#     check_requirements(
#         requirements=ROOT / "requirements.txt", exclude=("tensorboard", "thop")
#     )
#     run(opt, **vars(opt))


if __name__ == "__main__":
    data = 'yolov5/data/mapillary_no_color.yaml'
    run(data)
    # main()
