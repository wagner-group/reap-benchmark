"""Base attack class for all object detection models."""

from __future__ import annotations

from adv_patch_bench.attacks.base_attack import DetectorAttackModule
from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import BatchImageTensor, BatchMaskTensor


class NoAttackModule(DetectorAttackModule):
    """Base abstract class for attack on object detection models."""

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
        _ = rimg, batch_mode  # Unused
        return patch_mask
