"""DPatch attack for Detectron2 models."""

from __future__ import annotations

from typing import Any

from torch import nn

from adv_patch_bench.attacks.dpatch import dpatch_yolo
from adv_patch_bench.attacks.rp2 import rp2_yolo


class DPatchDinoAttack(dpatch_yolo.DPatchYoloAttack):
    """DPatch Attack for DINO models."""

    def __init__(
        self, attack_config: dict[str, Any], core_model: nn.Module, **kwargs
    ) -> None:
        """Initialize DPatchDinoAttack.

        Args:
            attack_config: Dictionary of attack params.
            core_model: Traget model to attack.
        """
        # Call RP2BaseAttack.__init__ instead of DPatchYolofAttack.__init__
        # since it does not refer to any model-specific attributes.
        super(rp2_yolo.RP2YoloAttack, self).__init__(
            attack_config, core_model, **kwargs
        )
        # Default DINO criterion params:
        # criterion=L(DINOCriterion)(
        #     num_classes="${..num_classes}",
        #     matcher=L(HungarianMatcher)(
        #         cost_class=2.0,
        #         cost_bbox=5.0,
        #         cost_giou=2.0,
        #         cost_class_type="focal_loss_cost",
        #         alpha=0.25,
        #         gamma=2.0,
        #     ),
        #     weight_dict={
        #         "loss_class": 1,
        #         "loss_bbox": 5.0,
        #         "loss_giou": 2.0,
        #         "loss_class_dn": 1,  # Not used during inference
        #         "loss_bbox_dn": 5.0,   # Not used during inference
        #         "loss_giou_dn": 2.0,   # Not used during inference
        #     },
        #     loss_class_type="focal_loss",
        #     alpha=0.25,
        #     gamma=2.0,
        #     two_stage_binary_cls=False,
        # ),

    def _on_enter_attack(self, **kwargs) -> None:
        self._is_training = self._core_model.training
        self._core_model.eval()
        self._core_model.attack_mode = True

    def _on_exit_attack(self, **kwargs) -> None:
        self._core_model.train(self._is_training)
        self._core_model.attack_mode = False
