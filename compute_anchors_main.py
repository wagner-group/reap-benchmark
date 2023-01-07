"""Compute anchors for YOLOv7.

Code is adapted from
https://github.com/jinfagang/yolov7_d2/blob/main/tools/compute_anchors.py.

Original warning: this anchor only useful when your YOLO input is not force
resized, which means your input image is padding at bottom and right without any
distortion. Otherwise this anchor is WRONG because we don't using forced resize
as input such as 608 or 512, we just using original image size!
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from detectron2.data import build_detection_train_loader
from tqdm import tqdm

import adv_patch_bench.dataloaders.detectron.util as data_util
from adv_patch_bench.dataloaders.detectron import mtsd_dataset_mapper
from adv_patch_bench.utils.argparse import reap_args_parser, setup_detectron_cfg

NUM_CLUSTERS = 9


def compute_iou(box, clusters):
    """Compute IOU between a box and the clusters."""
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        print("Box has no area")
        return 0
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """Compute the average IOU of all boxes with respect to all clusters."""
    return np.mean(
        [np.max(compute_iou(boxes[i], clusters)) for i in range(boxes.shape[0])]
    )


def run_kmeans_ious(boxes, k, dist=np.median):
    """Compute the k-means clustering of boxes."""
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - compute_iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters


def main():
    """Main function."""
    config: dict[str, dict[str, Any]] = reap_args_parser(
        True, is_gen_patch=False, is_train=True
    )
    config_base = config["base"]
    cfg = setup_detectron_cfg(config, is_train=True)
    # Register data. This has to be called by every process for some reason.
    data_util.register_dataset(config["base"])
    # pylint: disable=missing-kwoa,too-many-function-args
    data_loader = build_detection_train_loader(
        cfg,
        mapper=mtsd_dataset_mapper.MtsdDatasetMapper(
            cfg,
            is_train=True,
            img_size=config_base["img_size"],
            relight_method=config_base["reap_relight_method"],
            relight_percentile=config_base["reap_relight_percentile"],
        ),
    )

    bbox_sizes: list[list[float]] = []
    for _, data in enumerate(tqdm(data_loader)):
        for sample in data:
            bbox = sample["instances"].gt_boxes
            width = bbox.tensor[:, 2] - bbox.tensor[:, 0]
            height = bbox.tensor[:, 3] - bbox.tensor[:, 1]
            assert (width > 0).all() and (height > 0).all()
            sizes = torch.stack([height, width], dim=0).T.tolist()
            bbox_sizes.extend(sizes)

    print(f"Total number of bounding boxes: {len(bbox_sizes)}")
    out = run_kmeans_ious(bbox_sizes, k=NUM_CLUSTERS)
    print(f"Boxes: {out}")
    print(f"Accuracy: {avg_iou(bbox_sizes, out) * 100:.2f}%")

    final_anchors = np.around(out[:, 0] * out[:, 1], decimals=2).tolist()
    print(f"Before Sort area:\n {final_anchors}")
    idx = np.argsort(final_anchors)[::-1]
    print(f"After Sort area:\n {sorted(final_anchors)}")
    out = out[idx]
    print(f"Final anchor: {[list(np.array(a, dtype=np.int)) for a in out]}")


if __name__ == "__main__":
    main()
