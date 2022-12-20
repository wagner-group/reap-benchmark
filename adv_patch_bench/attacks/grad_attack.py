"""Base class of RP2 Attack."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from detectron2.utils.events import EventStorage
from torch import optim

from adv_patch_bench.attacks import base_attack
from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    BatchMaskTensor,
    Target,
)


class GradAttack(base_attack.DetectorAttackModule):
    """Base class of all gradient-based attacks."""

    def __init__(
        self,
        attack_config: dict[str, Any],
        core_model: torch.nn.Module,
        **kwargs,
    ):
        """Base class for gradient-based attack.

        Args:
            attack_config: Config dict for attacks.
            core_model: Target model to attack.
        """
        super().__init__(attack_config, core_model, **kwargs)
        # Attack parameters
        self._num_steps: int = attack_config["num_steps"]
        self._step_size: float = attack_config["step_size"]
        self._optimizer_name: str = attack_config["optimizer"]
        self._use_lr_schedule: bool = attack_config["use_lr_schedule"]
        self._num_eot: int = attack_config["num_eot"]
        self._lmbda: float = attack_config["lambda"]
        self._min_conf: float = attack_config["min_conf"]
        self._attack_mode: str = attack_config["attack_mode"].split("-")
        self._num_restarts: int = 1

        # Exponential moving average const
        self._ema_const: float = 0.9

        # TODO(feature): Allow more threat models
        self.targeted: bool = False

        # Use change of variable on delta with alpha and beta.
        # Mostly used with per-sign or real attack.
        self._use_var_change_ab: bool = "var_change_ab" in self._attack_mode
        if self._use_var_change_ab:
            # Does not work when num_eot > 1
            assert (
                self._num_eot == 1
            ), "When use_var_change_ab is used, num_eot can only be set to 1."

        self._ema_loss: float | None = None
        self._start_time = time.time()
        self._optimizer, self._lr_schedule = None, None

    def _reset_run(self, opt_var: torch.Tensor) -> None:
        """Reset each attack run."""
        self._ema_loss = None
        self._start_time = time.time()
        self._optimizer, self._lr_schedule = self._setup_opt(opt_var)

    @abstractmethod
    def compute_loss(
        self,
        delta: BatchImageTensor,
        adv_img: BatchImageTensor,
        adv_target: list[Target],
        # obj_class: int | None = None,
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
        raise NotImplementedError("self.compute_loss() not implemented!")

    @torch.enable_grad()
    def _run_one(
        self,
        rimg: render_image.RenderImage,
        z_delta: torch.Tensor,
        patch_mask: BatchMaskTensor,
        batch_mode: bool = False,
    ) -> BatchImageTensor:
        batch_size = len(z_delta)
        all_bg_idx: np.ndarray = np.arange(batch_size)

        # Run PGD on inputs for specified number of steps
        for step in range(self._num_steps):

            rimg_eot = rimg
            if not batch_mode:
                # Randomly select RenderImages to attack this step
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[: self._num_eot]
                rimg_eot = [rimg[i] for i in bg_idx]

            z_delta.requires_grad_()
            # Determine how perturbation is projected
            if self._use_var_change_ab:
                # TODO(feature): Does not work when num_eot > 1
                robj = rimg_eot[0].get_object()
                alpha, beta = robj.alpha, robj.beta
                delta = self._to_model_space(z_delta, beta, alpha + beta)
            else:
                delta = self._to_model_space(z_delta, 0, 1)

            # Load new adversarial patch to each RenderObject
            # adv_imgs = []
            # adv_targets = []
            # for rimg in rimg_eot:
            #     robj = rimg.get_object()
            #     robj.load_adv_patch(adv_patch=delta)
            #     # Apply adversarial patch to each RenderImage
            #     adv_img, adv_target = rimg.apply_objects()
            #     adv_img = rimg.post_process_image(adv_img)
            #     adv_imgs.append(adv_img)
            #     adv_targets.append(adv_target)

            adv_img, adv_target = rimg.apply_objects(delta, patch_mask)
            adv_img: BatchImageTensor = rimg.post_process_image(adv_img)

            loss: torch.Tensor = self.compute_loss(delta, adv_img, adv_target)
            loss.backward()
            z_delta = self._step_opt(z_delta)

            if self._lr_schedule is not None:
                self._lr_schedule.step(loss)
            self._print_loss(loss, step)

        # DEBUG
        # import os
        # for idx in range(self.num_eot):
        #     if not os.path.exists(f'tmp/{idx}/test_adv_img_{step}.png'):
        #         os.makedirs(f'tmp/{idx}/', exist_ok=True)
        #     torchvision.utils.save_image(adv_img[idx], f'tmp/{idx}/test_adv_img_{step}.png')
        return delta

    @torch.no_grad()
    def run(
        self,
        rimg: render_image.RenderImage,
        patch_mask: BatchMaskTensor,
        batch_mode: bool = False,
    ) -> BatchImageTensor:
        """Run gradient-based attack.

        Args:
            TODO(documentation)

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        self._on_enter_attack()
        device = patch_mask.device

        batch_size: int = 1
        if batch_mode:
            if patch_mask.ndim < 4:
                raise IndexError(
                    "patch_mask must have 4 dims if batch_mode is True!"
                )
            if rimg.num_objs != len(patch_mask):
                raise IndexError(
                    f"Number of objects in rimg ({rimg.num_objs}) must be equal"
                    f" to length of patch_mask ({len(patch_mask)}) if "
                    "batch_mode is True!"
                )
            batch_size = len(patch_mask)

        # Wrap with RenderImage -- already done

        # Create patches for all instances to attack

        # RenderImages apply patches in batch

        # Load patch_mask to all RenderObject first. This should be done once.
        # for i, rimg in enumerate(rimgs):
        #     robj = rimg.get_object()
        #     robj.load_adv_patch(
        #         patch_mask=patch_mask[i] if batch_mode else patch_mask
        #     )

        for _ in range(self._num_restarts):
            # Initialize adversarial perturbation
            z_delta: BatchImageTensor = torch.zeros(
                (batch_size, 3) + patch_mask.shape[-2:],
                device=device,
                dtype=torch.float32,
            )
            z_delta.uniform_(0, 1)
            # Set up optimizer
            self._reset_run(z_delta)
            # Run attack once
            with EventStorage():
                delta = self._run_one(
                    rimg, z_delta, patch_mask, batch_mode=batch_mode
                )

        # DEBUG
        # outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        # plot_images(adv_img.detach(), output_to_target(outt))

        self._on_exit_attack()
        # Return worst-case perturbed input logits
        return delta.detach()

    def _setup_opt(
        self, z_delta: BatchImageTensor
    ) -> tuple[optim.Optimizer | None, optim.lr_scheduler._LRScheduler | None]:
        # Set up optimizer
        if self._optimizer_name == "sgd":
            opt = optim.SGD([z_delta], lr=self._step_size, momentum=0.999)
        elif self._optimizer_name == "adam":
            opt = optim.Adam([z_delta], lr=self._step_size)
        elif self._optimizer_name == "rmsprop":
            opt = optim.RMSprop([z_delta], lr=self._step_size)
        elif self._optimizer_name == "pgd":
            opt = None
        else:
            raise NotImplementedError("Given optimizer not implemented.")

        lr_schedule = None
        if self._use_lr_schedule and opt is not None:
            lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=0.5,
                patience=int(self._num_steps / 10),
                threshold=1e-9,
                min_lr=self._step_size * 1e-6,
                verbose=self._verbose,
            )

        return opt, lr_schedule

    def _step_opt(self, z_delta: BatchImageTensor) -> BatchImageTensor:
        if self._optimizer == "pgd":
            grad = z_delta.grad.detach()
            grad = torch.sign(grad)
            z_delta = z_delta.detach() - self._step_size * grad
            z_delta.clamp_(0, 1)
        else:
            self._optimizer.step()
        return z_delta

    def _to_model_space(self, x, x_min, x_max):
        """Transforms an input from the attack space to the model space."""
        if "pgd" in self._attack_mode:
            return x

        # from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)

        # map from (-1, +1) to (min_, max_)
        a = (x_min + x_max) / 2
        b = (x_max - x_min) / 2
        x = x * b + a
        return x

    def _print_loss(self, loss: torch.Tensor, step: int) -> None:
        if self._ema_loss is None:
            self._ema_loss = loss.item()
        else:
            self._ema_loss = (
                self._ema_const * self._ema_loss
                + (1 - self._ema_const) * loss.item()
            )

        if step % 100 == 0 and self._verbose:
            print(
                f"step: {step:4d}  loss: {self._ema_loss:.4f}  "
                f"time: {time.time() - self._start_time:.2f}s"
            )
            self._start_time = time.time()
