"""DPatch attack for Detectron2 models."""

from __future__ import annotations

import copy
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
        self._use_only_final_loss = attack_config.get(
            "use_only_final_loss", False
        )
        self._weight_dict: dict[str, float] = {}

    def _on_enter_attack(self, **kwargs) -> None:
        self._is_training = self._core_model.training
        self._core_model.eval()
        core_module = self._core_model
        if hasattr(self._core_model, "module"):
            core_module = self._core_model.module
        core_module.attack_mode = True

        # Cache the original weight_dict
        self._weight_dict = copy.deepcopy(core_module.criterion.weight_dict)
        if self._use_only_final_loss:
            # Set all other loss weights to 0 except for actual pred loss
            for key in core_module.criterion.weight_dict.keys():
                if (
                    key.split("_")[-1].isnumeric()
                    or "dn" in key
                    or "enc" in key
                ):
                    core_module.criterion.weight_dict[key] = 0.0

    def _on_exit_attack(self, **kwargs) -> None:
        self._core_model.train(self._is_training)
        core_module = self._core_model
        if hasattr(self._core_model, "module"):
            core_module = self._core_model.module
        core_module.attack_mode = False
        core_module.criterion.weight_dict = self._weight_dict
