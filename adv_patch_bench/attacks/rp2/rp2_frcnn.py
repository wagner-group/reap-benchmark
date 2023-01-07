"""RP2 attack for Detectron2 models."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from detectron2 import structures
from torch import nn

from adv_patch_bench.attacks.rp2 import rp2_yolo
from adv_patch_bench.utils.types import BatchImageTensor, Target


class RP2AttackDetectron(rp2_yolo.RP2AttackYOLO):
    """RP2 Attack for Detectron2 models."""

    def __init__(
        self, attack_config: dict[str, Any], core_model: nn.Module, **kwargs
    ) -> None:
        """Initialize RP2AttackDetectron.

        TODO(feature): Currently, we assume that Detectron2 models are
        Faster R-CNN so loss function and params are specific to Faster R-CNN.
        We should implement attack on Faster R-CNN as subclass of Detectron2.

        Args:
            attack_config: Dictionary of attack params.
            core_model: Traget model to attack.
        """
        super().__init__(attack_config, core_model, **kwargs)

        detectron_config: dict[str, Any] = attack_config["detectron"]
        self._detectron_obj_const: float = detectron_config["obj_loss_const"]
        self._detectron_iou_thres: float = detectron_config["iou_thres"]
        # self._obj_class_only: bool = "obj_class_only" in self._attack_mode

        self._nms_thresh_orig = copy.deepcopy(
            core_model.proposal_generator.nms_thresh
        )
        self._post_nms_topk_orig = copy.deepcopy(
            core_model.proposal_generator.post_nms_topk
        )
        # self.nms_thresh = 0.9
        # self.post_nms_topk = {True: 5000, False: 5000}
        self._nms_thresh = self._nms_thresh_orig
        self._post_nms_topk = self._post_nms_topk_orig

    def _on_enter_attack(self, **kwargs) -> None:
        self._is_training = self._core_model.training
        self._core_model.eval()
        self._core_model.proposal_generator.nms_thresh = self._nms_thresh
        self._core_model.proposal_generator.post_nms_topk = self._post_nms_topk

    def _on_exit_attack(self, **kwargs) -> None:
        self._core_model.train(self._is_training)
        self._core_model.proposal_generator.nms_thresh = self._nms_thresh_orig
        self._core_model.proposal_generator.post_nms_topk = (
            self._post_nms_topk_orig
        )

    def _get_targets(
        self,
        inputs: list[dict[str, Any]],
        iou_thres: float = 0.1,
        score_thres: float = 0.1,
        use_correct_only: bool = False,
    ) -> tuple[structures.Boxes, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a set of targets to attack.

        Args:
            model: Model to attack.
            inputs: A list containing a single dataset_dict, transformed by
                a DatasetMapper.
            iou_thres: IoU threshold for matching predicted and ground-truth
                bouing boxes.
            score_thres: Predictions with class score less than score_thres are
                dropped.
            use_correct_only: Filter out predictions that are already incorrect.

        Returns:
            Matched gt target boxes, gt classes, predicted class logits, and
            predicted objectness logits.
        """
        device = self._core_model.device
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

        gt_boxes = [i["instances"].gt_boxes.to(device) for i in inputs]
        gt_classes = [i["instances"].gt_classes.to(device) for i in inputs]
        objectness_logits = [x.objectness_logits for x in proposals]

        proposal_boxes = [x.proposal_boxes for x in proposals]
        conds = _filter_positive_proposals(
            proposal_boxes,
            class_logits,
            gt_boxes,
            gt_classes,
            iou_thres=iou_thres,
            score_thres=score_thres,
            use_correct_only=use_correct_only,
        )
        outputs = []
        for i, (paired_gt_classes, cond) in enumerate(conds):
            outputs.append(
                [
                    paired_gt_classes.to(device),
                    class_logits[i][cond],
                    objectness_logits[i][cond],
                ]
            )
            num_pairs = len(paired_gt_classes)
            assert (
                num_pairs == len(outputs[-1][1]) == len(outputs[-1][2])
            ), f"Output shape mismatch: {[len(o) for o in outputs[-1]]}!"
        return outputs

    def _generate_proposal(
        self,
        images: structures.ImageList,
        features: dict[str, torch.Tensor],
        compute_loss: bool = True,
        gt_instances: list[structures.Instances] | None = None,
    ) -> tuple[list[structures.Instances], dict[str, torch.Tensor]]:
        """Generates proposal and computes RPN loss.

        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of
                `Instances`s. Each `Instances` stores ground-truth instances for
                the corresponding image.

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

    def _loss_func(
        self,
        adv_imgs: BatchImageTensor,
        adv_targets: list[Target],
        # obj_class: int | None = None,
    ) -> torch.Tensor:
        """Compute loss for Faster R-CNN models on detectron2.

        Args:
            adv_img: Image to compute loss on.
            adv_target: Target label to compute loss on.
            obj_class: Target object class. Usually ground-truth label for
                untargeted attack, and target class for targeted attack.

        Returns:
            Loss for attacker to minimize.
        """
        # NOTE: IoU threshold for ROI is 0.5 and for RPN is 0.7
        inputs = []
        for img, target in zip(adv_imgs, adv_targets):
            target["image"] = img
            inputs.append(target)
        # pylint: disable=unbalanced-tuple-unpacking
        outputs = self._get_targets(
            inputs,
            iou_thres=self._detectron_iou_thres,
            score_thres=self._min_conf,
            use_correct_only=False,
        )

        # DEBUG
        # import cv2
        # from detectron2.utils.visualizer import Visualizer
        # from detectron2.data import MetadataCatalog
        # with torch.no_grad():
        #     idx = 0
        #     metadata[idx]['height'], metadata[idx]['width'] = adv_img.shape[2:]
        #     outputs = self.core_model(metadata)[idx]
        #     instances = outputs["instances"]
        #     mask = instances.scores > 0.5
        #     instances = instances[mask]
        #     self.metadata = MetadataCatalog.get('mapillary_combined')
        #     img = metadata[idx]['image'].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
        #     v = Visualizer(img, self.metadata, scale=0.5)
        #     vis_og = v.draw_instance_predictions(instances.to('cpu')).get_image()
        #     cv2.imwrite('temp_pred.png', vis_og[:, :, ::-1])
        #     metadata[idx]['annotations'] = [{
        #         'bbox': metadata[idx]['instances'].gt_boxes.tensor[0].tolist(),
        #         'category_id': metadata[idx]['instances'].gt_classes.item(),
        #         'bbox_mode': metadata[idx]['annotations'][0]['bbox_mode'],
        #     }]
        #     vis_gt = v.draw_dataset_dict(metadata[0]).get_image()
        #     cv2.imwrite('temp_gt.png', vis_gt[:, :, ::-1])
        #     print('ok')
        # import pdb
        # pdb.set_trace()

        # Loop through each EoT image
        loss: torch.Tensor = torch.zeros(1, device=adv_imgs.device)
        for tgt_lb, tgt_log, obj_log in outputs:
            # Filter obj_class
            # if self._obj_class_only:
            #     # Focus attack on prediction of `obj_class` only
            #     idx = obj_class == tgt_lb
            #     tgt_lb, tgt_log, obj_log = (
            #         tgt_lb[idx],
            #         tgt_log[idx],
            #         obj_log[idx],
            #     )
            # else:
            #     tgt_lb = torch.zeros_like(tgt_lb) + obj_class
            # If there's no matched gt/prediction, then attack already succeeds.
            # TODO(feature): Appearing or misclassification attacks
            target_loss: torch.Tensor = torch.zeros_like(loss)
            obj_loss: torch.Tensor = torch.zeros_like(loss)
            if len(tgt_log) > 0 and len(tgt_lb) > 0:
                # Ignore the background class on tgt_log
                target_loss = F.cross_entropy(tgt_log, tgt_lb, reduction="mean")
            if len(obj_log) > 0 and self._detectron_obj_const != 0:
                obj_lb = torch.ones_like(obj_log)
                obj_loss = F.binary_cross_entropy_with_logits(
                    obj_log, obj_lb, reduction="mean"
                )
            loss += target_loss + self._detectron_obj_const * obj_loss
        return -loss


def _filter_positive_proposals(
    proposal_boxes: list[structures.Boxes],
    class_logits: list[torch.Tensor],
    gt_boxes: list[structures.Boxes],
    gt_classes: list[torch.Tensor],
    **kwargs,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """See _filter_positive_proposals_single()."""
    outputs = []
    for inpt in zip(proposal_boxes, class_logits, gt_boxes, gt_classes):
        outputs.append(_filter_positive_proposals_single(*inpt, **kwargs))
    return outputs


def _filter_positive_proposals_single(
    proposal_boxes: structures.Boxes,
    class_logits: torch.Tensor,
    gt_boxes: structures.Boxes,
    gt_classes: torch.Tensor,
    iou_thres: float = 0.1,
    score_thres: float = 0.1,
    use_correct_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter for desired targets for the attack.

    Args:
        proposal_boxes: Proposal boxes directly from RPN.
        scores: Softmaxed scores for each proposal box.
        gt_boxes: Ground truth boxes.
        gt_classes: Ground truth classes.

    Returns:
        Filtered target boxes and corresponding class labels.
    """
    n_proposals: int = len(proposal_boxes)
    device = class_logits.device

    proposal_gt_ious: torch.Tensor = structures.pairwise_iou(
        proposal_boxes, gt_boxes
    ).to(device)

    # Pair each proposed box in proposal_boxes with a ground-truth box in
    # gt_boxes, i.e., find ground-truth box with highest IoU.
    # IoU with paired gt_box, idx of paired gt_box
    paired_ious, paired_gt_idx = proposal_gt_ious.max(dim=1)

    # Filter for IoUs > iou_thres
    iou_cond = paired_ious >= iou_thres

    # Get class of paired gt_box
    gt_classes_repeat = gt_classes.repeat(n_proposals, 1)
    idx = torch.arange(n_proposals, device=device)
    paired_gt_idx = paired_gt_idx.to(device)
    paired_gt_classes = gt_classes_repeat[idx, paired_gt_idx]

    cond = iou_cond
    if use_correct_only:
        # Filter for score of proposal > score_thres
        # Get scores of corresponding class
        scores = F.softmax(class_logits, dim=-1)
        paired_scores = scores[idx, paired_gt_classes]
        score_cond = paired_scores >= score_thres
        # Filter for positive proposals and their corresponding gt labels
        cond = iou_cond & score_cond

    return paired_gt_classes[cond], cond
