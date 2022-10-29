"""Base attack class for all object detection models."""

import abc
from typing import Any, Dict

import torch
import torch.nn as nn
from adv_patch_bench.utils.types import ImageTensor


class DetectorAttackModule(nn.Module):
    """Base abstract class for attack on object detection models."""

    def __init__(
        self,
        attack_config: Dict[str, Any],
        core_model: torch.nn.Module,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize DetectorAttackModule.

        Args:
            attack_config: Config dict for attacks.
            core_model: Target model to attack.
            verbose: Whether to log messages during attack.
        """
        super().__init__()
        self._core_model: torch.nn.Module = core_model
        self._verbose: bool = verbose

    def _on_enter_attack(self, **kwargs) -> None:
        """Method called at the begining of the attack call."""
        self.is_training = self._core_model.training
        self._core_model.eval()

    def _on_exit_attack(self, **kwargs) -> None:
        """Method called at the end of the attack call."""
        self._core_model.train(self.is_training)

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> ImageTensor:
        """Run attack.

        Returns:
            Adversarial patch.
        """
        raise NotImplementedError("run() must be implemented!")
