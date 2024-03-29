"""DPatch attack for Detectron2 models."""

from __future__ import annotations

from typing import Any

import torch
from detectron2 import structures

from adv_patch_bench.attacks.rp2 import rp2_faster_rcnn
from adv_patch_bench.utils.types import BatchImageTensor, Target


class DPatchFasterRCNNAttack(rp2_faster_rcnn.RP2FasterRCNNAttack):
    """DPatch Attack for Detectron2 models."""

    def _loss_func(
        self,
        adv_imgs: BatchImageTensor,
        adv_targets: list[Target],
    ) -> torch.Tensor:
        """Compute DPatch loss for Faster R-CNN models.

        Args:
            adv_img: Image to compute loss on.
            adv_target: Target label to compute loss on.

        Returns:
            Loss for attacker to minimize.
        """
        inputs: list[dict[str, Any]] = adv_targets
        instances: list[structures.Instances] = []
        device = self._core_model.device
        core_model = self._core_model
        if hasattr(self._core_model, "module"):
            core_model = self._core_model.module

        for i, inpt in enumerate(inputs):
            inpt["image"] = adv_imgs[i]
            instances.append(inpt["instances"].to(device))

        # Get features
        images = core_model.preprocess_image(inputs)
        features = core_model.backbone(images.tensor)

        # Get bounding box proposals
        proposals, proposal_losses = self._generate_proposal(
            images,
            features,
            compute_loss=True,
            gt_instances=instances,
        )

        # Get proposal boxes' classification scores
        _, roi_losses = self._get_roi_heads_predictions(
            features,
            proposals,
            compute_loss=True,
            gt_instances=instances,
        )

        # TODO(feature): Custom weights on losses
        loss = (
            proposal_losses["loss_rpn_cls"]
            + proposal_losses["loss_rpn_loc"]
            + roi_losses["loss_cls"]
            + roi_losses["loss_box_reg"]
        )

        return -loss
