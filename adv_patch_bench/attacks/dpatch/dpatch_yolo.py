"""DPatch attack for Detectron2 models."""

from __future__ import annotations

from typing import Any

import torch
from detectron2 import structures

from adv_patch_bench.attacks.rp2 import rp2_yolo
from adv_patch_bench.utils.types import BatchImageTensor, Target


class DPatchYoloAttack(rp2_yolo.RP2YoloAttack):
    """DPatch Attack for Detectron2 models."""

    def _loss_func(
        self,
        adv_imgs: BatchImageTensor,
        adv_targets: list[Target],
    ) -> torch.Tensor:
        """Compute DPatch loss for YOLO models.

        Args:
            adv_imgs: Batch of images to compute loss on.
            adv_targets: Target labels to compute loss on.

        Returns:
            Loss for attacker to **minimize**.
        """
        inputs: list[dict[str, Any]] = adv_targets
        instances: list[structures.Instances] = []
        device = self._core_model.device
        for i, inpt in enumerate(inputs):
            inpt["image"] = adv_imgs[i]
            instances.append(inpt["instances"].to(device))

        output = self._core_model(inputs)
        if isinstance(output, dict):
            losses = output
        else:
            losses = output[-1]

        # TODO(enhancement): Add loss weights (lambdas) are set in model init
        # with the following configs:
        # self.lambda_xy = cfg.MODEL.YOLO.LOSS.LAMBDA_XY
        # self.lambda_wh = cfg.MODEL.YOLO.LOSS.LAMBDA_WH
        # self.lambda_conf = cfg.MODEL.YOLO.LOSS.LAMBDA_CONF
        # self.lambda_cls = cfg.MODEL.YOLO.LOSS.LAMBDA_CLS
        # self.lambda_iou = cfg.MODEL.YOLO.LOSS.LAMBDA_IOU
        loss = sum(losses.values())

        return -loss
