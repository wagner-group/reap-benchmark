"""Base attack class for all object detection models."""

from __future__ import annotations

import abc
from typing import Any

import torch
from torch import nn

from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import BatchImageTensor, BatchMaskTensor


class DetectorAttackModule(nn.Module):
    """Base abstract class for attack on object detection models."""

    def __init__(
        self,
        attack_config: dict[str, Any],
        core_model: torch.nn.Module,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Initialize DetectorAttackModule.

        Args:
            attack_config: Config dict for attacks.
            core_model: Target model to attack.
            verbose: Whether to log messages during attack.
        """
        super().__init__()
        _ = attack_config, kwargs  # Unused
        self._core_model: torch.nn.Module = core_model
        self._verbose: bool = verbose
        self._is_training: bool = False

    def _on_enter_attack(self, **kwargs) -> None:
        """Method called at the begining of the attack call."""
        _ = kwargs  # Unused
        self._is_training = self._core_model.training
        self._core_model.eval()

    def _on_exit_attack(self, **kwargs) -> None:
        """Method called at the end of the attack call."""
        _ = kwargs  # Unused
        self._core_model.train(self._is_training)

    @abc.abstractmethod
    def run(
        self,
        rimg: render_image.RenderImage,
        patch_mask: BatchMaskTensor,
        batch_mode: bool = False,
    ) -> BatchImageTensor:
        """Run attack.

        Returns:
            Adversarial patch.
        """
        raise NotImplementedError("run() must be implemented!")

    def forward(self, *args, **kwargs) -> BatchImageTensor:
        """Run attack with forward."""
        return self.run(*args, **kwargs)
