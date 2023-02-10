"""RP2 Attack for YOLO models."""

from __future__ import annotations

import copy
from typing import Any

import torch
from detectron2 import structures
from torch import nn

from adv_patch_bench.attacks.rp2 import rp2_base


class RP2YoloAttack(rp2_base.RP2BaseAttack):
    """RP2 Attack for YOLO models."""

    def __init__(
        self, attack_config: dict[str, Any], core_model: nn.Module, **kwargs
    ) -> None:
        """Initialize RP2YoloAttack.

        Args:
            attack_config: Dictionary of attack params.
            core_model: Traget model to attack.
        """
        super().__init__(attack_config, core_model, **kwargs)
        if hasattr(core_model, "module"):
            core_model = core_model.module
        self._nms_thres_orig = copy.deepcopy(core_model.nms_threshold)
        self._conf_thres_orig = copy.deepcopy(core_model.conf_threshold)
        # loss_evaluators[0] is YOLOHead
        self._iou_thres_orig = copy.deepcopy(
            core_model.loss_evaluators[0].iou_threshold
        )
        if self._nms_thres is None:
            self._nms_thres = self._nms_thres_orig
        if self._min_conf is None:
            self._min_conf = self._conf_thres_orig
        if self._iou_thres is None:
            self._iou_thres = self._iou_thres_orig

    def _on_enter_attack(self, **kwargs) -> None:
        self._is_training = self._core_model.training
        self._core_model.eval()
        if hasattr(self._core_model, "module"):
            core_model = self._core_model.module
        else:
            core_model = self._core_model
        core_model.attack_mode = True
        core_model.nms_threshold = self._nms_thres
        core_model.conf_threshold = self._min_conf
        core_model.loss_evaluators[0].iou_threshold = self._iou_thres

    def _on_exit_attack(self, **kwargs) -> None:
        self._core_model.train(self._is_training)
        if hasattr(self._core_model, "module"):
            core_model = self._core_model.module
        else:
            core_model = self._core_model
        core_model.attack_mode = False
        core_model.nms_threshold = self._nms_thres_orig
        core_model.conf_threshold = self._conf_thres_orig
        core_model.loss_evaluators[0].iou_threshold = self._iou_thres_orig

    def _get_targets(
        self,
        inputs: list[dict[str, Any]],
        use_correct_only: bool = False,
    ) -> tuple[structures.Boxes, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a set of targets to attack.

        Args:
            inputs: A list containing a single dataset_dict, transformed by
                a DatasetMapper.
            use_correct_only: Filter out predictions that are already incorrect.

        Returns:
            Matched gt target boxes, gt classes, predicted class logits, and
            predicted objectness logits.
        """
        # losses contain total_loss, iou_loss, conf_loss, cls_loss, l1_loss
        results, _, _ = self._core_model(inputs)

        pred_boxes = [result["instances"].pred_boxes for result in results]
        class_logits = [result["instances"].cls_logits for result in results]
        obj_logits = [result["instances"].obj_logits for result in results]
        gt_boxes = [tgt["instances"].gt_boxes for tgt in inputs]
        gt_classes = [tgt["instances"].gt_classes for tgt in inputs]
        paired_outputs = self._pair_gt_proposals(
            pred_boxes,
            class_logits,
            obj_logits,
            gt_boxes,
            gt_classes,
            use_correct_only=use_correct_only,
        )
        return paired_outputs
