"""RP2 attack for Detectron2 models."""

from __future__ import annotations

import copy
from typing import Any

import torch
from detectron2 import structures
from torch import nn

from adv_patch_bench.attacks.rp2 import rp2_base


class RP2FasterRCNNAttack(rp2_base.RP2BaseAttack):
    """RP2 Attack for Detectron2 models."""

    def __init__(
        self, attack_config: dict[str, Any], core_model: nn.Module, **kwargs
    ) -> None:
        """Initialize RP2FasterRCNNAttack.

        Args:
            attack_config: Dictionary of attack params.
            core_model: Traget model to attack.
        """
        super().__init__(attack_config, core_model, **kwargs)
        # NOTE: Score threshold is used in rp2_base.RP2BaseAttack
        self._nms_thres_orig = copy.deepcopy(
            core_model.proposal_generator.nms_thresh
        )
        # RPN iou_thres: [-inf, 0.3, 0.7, inf]
        self._iou_thres_rpn_orig = copy.deepcopy(
            core_model.proposal_generator.anchor_matcher.thresholds
        )
        # ROI heads iou_thres: [-inf, 0.5, inf]
        self._iou_thres_roi_orig = copy.deepcopy(
            core_model.roi_heads.proposal_matcher.thresholds
        )
        self._post_nms_topk_orig = copy.deepcopy(
            core_model.proposal_generator.post_nms_topk
        )
        # self.nms_thresh = 0.9
        # self.post_nms_topk = {True: 5000, False: 5000}
        self._post_nms_topk = self._post_nms_topk_orig
        if self._nms_thres is None:
            self._nms_thres = self._nms_thres_orig
        if self._iou_thres is None:
            self._iou_thres_rpn = self._iou_thres_rpn_orig
            self._iou_thres_roi = self._iou_thres_roi_orig
        else:
            # Replace foreground threshold with iou_thres for RPN
            self._iou_thres_rpn = [
                self._iou_thres_rpn_orig[0],
                min(self._iou_thres_rpn_orig[1], self._iou_thres - 1e-3),
                self._iou_thres,
                self._iou_thres_rpn_orig[3],
            ]
            # Replace threshold with iou_thres for ROI heads
            self._iou_thres_roi = [
                self._iou_thres_roi_orig[0],
                self._iou_thres,
                self._iou_thres_roi_orig[2],
            ]

    def _on_enter_attack(self, **kwargs) -> None:
        self._is_training = self._core_model.training
        self._core_model.eval()
        self._core_model.proposal_generator.nms_thresh = self._nms_thresh
        self._core_model.proposal_generator.post_nms_topk = self._post_nms_topk
        self._core_model.roi_heads.proposal_matcher.thresholds = (
            self._iou_thres_roi
        )
        self._core_model.proposal_generator.anchor_matcher.thresholds = (
            self._iou_thres_rpn
        )

    def _on_exit_attack(self, **kwargs) -> None:
        self._core_model.train(self._is_training)
        self._core_model.proposal_generator.nms_thresh = self._nms_thres_orig
        self._core_model.proposal_generator.post_nms_topk = (
            self._post_nms_topk_orig
        )
        self._core_model.roi_heads.proposal_matcher.thresholds = (
            self._iou_thres_roi_orig
        )
        self._core_model.proposal_generator.anchor_matcher.thresholds = (
            self._iou_thres_rpn_orig
        )

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
        images = self._core_model.preprocess_image(inputs)

        # Get features
        features = self._core_model.backbone(images.tensor)

        # Get bounding box proposals. For API, see
        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/proposal_generator/rpn.py#L431
        proposals, _ = self._generate_proposal(
            images,
            features,
            compute_loss=False,
        )

        # Get proposal boxes' classification scores
        # See detectron2.modeling.roi_heads.roi_heads.StandardROIHeads._forward_box
        predictions, _ = self._get_roi_heads_predictions(
            features,
            proposals,
            compute_loss=False,
        )

        # We can get softmax scores for a single image, [n_proposals, n_classes + 1]
        # scores = model.roi_heads.box_predictor.predict_probs(predictions, proposals)[0]
        # Instead, we want to get logit scores without softmax. For API, see
        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L547
        class_logits, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        # NOTE: class_logits dim [[1000, num_classes + 1], ...]
        class_logits = class_logits.split(num_inst_per_image, dim=0)

        gt_boxes = [i["instances"].gt_boxes.to(self._device) for i in inputs]
        gt_classes = [
            i["instances"].gt_classes.to(self._device) for i in inputs
        ]
        objectness_logits = [x.objectness_logits for x in proposals]

        proposal_boxes = [x.proposal_boxes for x in proposals]
        paired_outputs = self._pair_gt_proposals(
            proposal_boxes,
            class_logits,
            objectness_logits,
            gt_boxes,
            gt_classes,
            use_correct_only=use_correct_only,
        )
        return paired_outputs

    def _generate_proposal(
        self,
        images: structures.ImageList,
        features: dict[str, torch.Tensor],
        compute_loss: bool = True,
        gt_instances: list[structures.Instances] | None = None,
    ) -> tuple[list[structures.Instances], dict[str, torch.Tensor]]:
        """Generates proposal and computes RPN loss.

        Args:
            images: input images of length `N`
            features: input data as a mapping from feature map name to tensor.
                Axis 0 represents the number of images `N` in the input data;
                axes 1-3 are channels, height, and width, which may vary between
                feature maps (e.g., if a feature pyramid is used).
            gt_instances: a length `N` list of `Instances`s. Each `Instances`
                stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes",
                "objectness_logits".
            loss: dict[Tensor] if compute_loss is True else None.
        """
        rpn: nn.Module = self._core_model.proposal_generator
        features = [features[f] for f in rpn.in_features]
        anchors = rpn.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = rpn.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) ->
            # (N, Hi*Wi*A, B)
            x.view(
                x.shape[0],
                -1,
                rpn.anchor_generator.box_dim,
                x.shape[-2],
                x.shape[-1],
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if compute_loss:
            assert (
                gt_instances is not None
            ), "RPN requires gt_instances to compute loss."
            gt_labels, gt_boxes = rpn.label_and_sample_anchors(
                anchors, gt_instances
            )
            losses = rpn.losses(
                anchors,
                pred_objectness_logits,
                gt_labels,
                pred_anchor_deltas,
                gt_boxes,
            )
        else:
            losses = {}

        # Decode all the predicted box regression deltas to proposals. Find the top
        # proposals by applying NMS and removing boxes that are too small. Returns
        # proposals (list[Instances]): list of N Instances. The i-th Instances
        # stores post_nms_topk object proposals for image i, sorted by their
        # objectness score in descending order.
        proposals = rpn.predict_proposals(
            anchors,
            pred_objectness_logits,
            pred_anchor_deltas,
            images.image_sizes,
        )
        return proposals, losses

    def _get_roi_heads_predictions(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[structures.Instances],
        compute_loss: bool = True,
        gt_instances: list[structures.Instances] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """Get predictions and losses from ROI heads.

        See detectron2.modeling.roi_heads.roi_heads.StandardROIHeads._forward_box.

        Args:
            roi_heads: ROI head.
            features: input data as a mapping from feature map name to tensor.
                Axis 0 represents the number of images `N` in the input data; axes
                1-3 are channels, height, and width, which may vary between feature
                maps (e.g., if a feature pyramid is used).
            proposals: length `N` list of `Instances`. The i-th `Instances` contains
                object proposals for the i-th input image, with fields
                "proposal_boxes" and "objectness_logits".
            compute_loss: If True, also computes ROI losses.

        Returns:
            predictions: First tensor: shape (N,K+1), scores for each of the N box.
                Each row contains the logit scores for K object categories and 1
                background class. Second tensor: bounding box regression deltas for
                each box. Shape is shape (N,Kx4) or (N,4) for class-agnostic
                regression.
            losses: ROI losses if compute_loss is True.
        """
        roi_heads = self._core_model.roi_heads
        features = [features[f] for f in roi_heads.box_in_features]
        if gt_instances is not None:
            proposals = roi_heads.label_and_sample_proposals(
                proposals, gt_instances
            )
        box_features = roi_heads.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = roi_heads.box_head(box_features)
        # Box predictor is FastRCNNOutputLayers. predictions contain logit scores
        # and proposal_deltas.
        predictions = roi_heads.box_predictor(box_features)
        del box_features
        if compute_loss:
            losses = roi_heads.box_predictor.losses(predictions, proposals)
        else:
            losses = {}
        return predictions, losses
