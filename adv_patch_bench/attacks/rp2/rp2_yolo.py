"""RP2 Attack for YOLO models."""

from __future__ import annotations

import torch

from adv_patch_bench.attacks import grad_attack
from adv_patch_bench.utils.types import ImageTensor, Target


class RP2AttackYOLO(grad_attack.GradAttack):
    """RP2 Attack for YOLO models."""

    def _loss_func(
        self,
        adv_imgs: list[ImageTensor],
        adv_targets: list[Target],
        obj_class: int | None = None,
    ) -> torch.Tensor:
        """Compute loss for YOLO models.

        TODO(documentation)
        """
        # Compute logits, loss, gradients
        out, _ = self._core_model(adv_imgs, val=True)
        conf = out[:, :, 4:5] * out[:, :, 5:]
        conf, labels = conf.max(-1)
        if obj_class is not None:
            loss = 0
            # Loop over EoT batch
            for c, label in zip(conf, labels):
                c_l = c[label == obj_class]
                if c_l.size(0) > 0:
                    # Select prediction from box with max confidence and ignore
                    # ones with already low confidence
                    # loss += c_l.max().clamp_min(self.min_conf)
                    loss += c_l.clamp_min(self._min_conf).sum()
            loss /= self._num_eot
        else:
            # loss = conf.max(1)[0].clamp_min(self.min_conf).mean()
            loss = conf.clamp_min(self._min_conf).sum()
            loss /= self._num_eot
        return loss

    def compute_loss(
        self,
        delta: ImageTensor,
        adv_imgs: list[ImageTensor],
        adv_targets: list[Target],
        obj_class: int | None = None,
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
        loss: torch.Tensor = self._loss_func(adv_imgs, adv_targets, obj_class)
        reg: torch.Tensor = (
            delta[:, :-1, :] - delta[:, 1:, :]
        ).abs().mean() + (delta[:, :, :-1] - delta[:, :, 1:]).abs().mean()
        loss += self._lmbda * reg
        return loss
