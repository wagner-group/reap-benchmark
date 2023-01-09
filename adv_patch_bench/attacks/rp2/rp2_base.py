"""RP2 attack for Detectron2 models."""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn.functional as F
from detectron2 import structures
from torch import nn

from adv_patch_bench.attacks import grad_attack
from adv_patch_bench.utils.types import BatchImageTensor, Target


class RP2BaseAttack(grad_attack.GradAttack):
    """RP2 Attack for Detectron2 models."""

    def __init__(
        self, attack_config: dict[str, Any], core_model: nn.Module, **kwargs
    ) -> None:
        """Initialize RP2BaseAttack.

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

    def compute_loss(
        self,
        delta: BatchImageTensor,
        adv_imgs: BatchImageTensor,
        adv_targets: list[Target],
    ) -> torch.Tensor:
        """Compute loss on perturbed image.

        Args:
            delta: Adversarial patch.
            adv_img: Perturbed image to compute loss on.
            adv_target: Target label to compute loss on.
            obj_class: Target object class. Usually ground-truth label for
                untargeted attack, and target class for targeted attack.

        Returns:
            Loss for attacker to minimize.
        """
        loss: torch.Tensor = self._loss_func(adv_imgs, adv_targets)
        reg: torch.Tensor = (
            delta[:, :-1, :] - delta[:, 1:, :]
        ).abs().mean() + (delta[:, :, :-1] - delta[:, :, 1:]).abs().mean()
        loss += self._lmbda * reg
        return loss

    @abc.abstractmethod
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
        _ = inputs, use_correct_only  # Unused
        raise NotImplementedError("_get_targets() has to be implemented.")

    def _loss_func(
        self,
        adv_imgs: BatchImageTensor,
        adv_targets: list[Target],
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
        outputs = self._get_targets(inputs, use_correct_only=False)

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

    def _pair_gt_proposals(
        self,
        proposal_boxes: list[structures.Boxes],
        class_logits: torch.Tensor | list[torch.Tensor],
        objectness_logits: torch.Tensor | list[torch.Tensor],
        gt_boxes: list[structures.Boxes],
        gt_classes: list[torch.Tensor],
        use_correct_only: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """See _filter_positive_proposals_single().

        # TODO(documentation): Add docstring.
        """
        pairs = []
        for inpt in zip(proposal_boxes, class_logits, gt_boxes, gt_classes):
            pairs.append(
                _filter_positive_proposals_single(
                    *inpt,
                    iou_thres=self._detectron_iou_thres,
                    score_thres=self._min_conf,
                    use_correct_only=use_correct_only,
                )
            )
        paired_outputs = []
        for i, (paired_gt_classes, paired_idx) in enumerate(pairs):
            paired_outputs.append(
                [
                    paired_gt_classes.to(self._device),
                    class_logits[i][paired_idx],
                    objectness_logits[i][paired_idx],
                ]
            )
            num_pairs = len(paired_gt_classes)
            assert all(
                num_pairs == len(output) for output in paired_outputs[-1]
            ), f"Output shape mismatch: {[len(o) for o in paired_outputs[-1]]}!"
        return paired_outputs


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
