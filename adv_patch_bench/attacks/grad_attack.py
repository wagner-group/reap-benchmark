"""Base class of RP2 Attack."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

import torch
from detectron2.utils.events import EventStorage
from torch import optim

from adv_patch_bench.attacks import base_attack
from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    BatchMaskTensor,
    ImageTensor,
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
        bg_idx: torch.Tensor | None = None
        delta = z_delta
        # Initialize religting params
        if self._use_var_change_ab or "pgd" in self._optimizer_name:
            beta = rimg.tf_params["beta"].clone()
            half_alpha = rimg.tf_params["alpha"] / 2
            # Temporary set alphas and betas to 1 and 0 so patch does not get
            # rescaled again
            rimg.tf_params["beta"].zero_()
            rimg.tf_params["alpha"].fill_(1)
        else:
            beta = torch.zeros_like(rimg.tf_params["beta"])
            half_alpha = torch.ones_like(rimg.tf_params["beta"]) / 2

        # Run PGD on inputs for specified number of steps
        for step in range(self._num_steps):
            # Randomly select RenderImages to attack this step
            if not batch_mode and self._num_eot < rimg.num_objs:
                bg_idx = torch.randperm(rimg.num_objs, device=z_delta.device)
                bg_idx = bg_idx[: self._num_eot]
                # DEBUG: Fix bg_idx to debug
                # bg_idx = torch.zeros(1, device=z_delta.device, dtype=torch.long)

            if "pgd" in self._optimizer_name:
                cur_half_alpha, cur_beta = self._select_tensors(
                    bg_idx, half_alpha, beta
                )
                delta = delta.detach()
                delta = torch.maximum(delta, cur_beta)
                delta = torch.minimum(delta, cur_beta + 2 * cur_half_alpha)
                delta.requires_grad_()
            else:
                self._optimizer.zero_grad()
                z_delta.requires_grad_()
                delta = self._to_input_space(z_delta, half_alpha, beta, bg_idx)

            # Apply patch with transforms and compute loss
            adv_img, adv_target = rimg.apply_objects(
                adv_patch=delta, patch_mask=patch_mask, obj_indices=bg_idx
            )
            adv_img: BatchImageTensor = rimg.post_process_image(adv_img)
            loss: torch.Tensor = self.compute_loss(delta, adv_img, adv_target)
            loss.backward()

            # Update perturbation
            if "pgd" in self._optimizer_name:
                grad = delta.grad.detach()
                grad = torch.sign(grad)
                delta.detach_()
                delta -= self._step_size * grad
            else:
                self._optimizer.step()

            if self._lr_schedule is not None:
                self._lr_schedule.step(loss)
            self._print_loss(loss, step)

        if self._use_var_change_ab:
            rimg.tf_params["beta"] = beta
            rimg.tf_params["alpha"] = half_alpha * 2
        return delta

    @torch.no_grad()
    def run(
        self,
        rimg: render_image.RenderImage,
        patch_mask: BatchMaskTensor,
        batch_mode: bool = False,
        init_adv_patch: list[ImageTensor | None] | None = None,
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
                    f"Number of objects in rimg ({rimg.num_objs}) must be "
                    f"equal to length of patch_mask ({len(patch_mask)}) if "
                    "batch_mode is True!"
                )
            batch_size = len(patch_mask)
        else:
            if "pgd" in self._optimizer_name and rimg.num_objs > 1:
                # TODO(feature): We can allow this if alphas and betas are the
                # same for all objects.
                raise ValueError(
                    "PGD optimizer cannot be used in non-batch mode with more "
                    f"than one obects ({rimg.num_objs}). PGD clipping should "
                    "be fixed so it cannot depend on multiple objects."
                )

        if init_adv_patch is None:
            init_adv_patch = [None] * batch_size
        assert len(init_adv_patch) == batch_size

        for _ in range(self._num_restarts):
            # Initialize adversarial perturbation
            z_delta: BatchImageTensor = torch.zeros(
                (batch_size, 3) + patch_mask.shape[-2:],
                device=device,
                dtype=torch.float32,
            )
            z_delta.uniform_(0 if "pgd" in self._optimizer_name else -1, 1)

            for i, init_patch in enumerate(init_adv_patch):
                if init_patch is not None:
                    z_delta[i] = self._to_opt_space(init_patch.to(device))

            if not batch_mode:
                z_delta = z_delta.expand(self._num_eot, -1, -1, -1)
                patch_mask = patch_mask.expand(self._num_eot, -1, -1, -1)

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

    @staticmethod
    def _select_tensors(indices, *args):
        if indices is None:
            return args
        return [tensor.index_select(0, indices) for tensor in args]

    def _to_input_space(
        self,
        images: torch.Tensor,
        half_alpha: torch.Tensor,
        beta: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transforms an input from the attack space to the model space."""
        half_alpha, beta = self._select_tensors(indices, half_alpha, beta)
        # from (-inf, +inf) to (-1, +1)
        images = torch.tanh(images)
        # map from (-1, +1) to (low, high)
        # Need to copy images here since torch.tanh needs output to compute grad
        images = images * half_alpha
        images.add_(half_alpha).add_(beta)
        return images

    def _to_opt_space(self, images):
        if "pgd" in self._optimizer_name:
            return images
        EPS = 1e-6
        assert (images >= 0 & images <= 1).all()
        images = images * 2 - 1
        images.clamp_(-1 + EPS, 1 - EPS)
        images.arctanh_()
        return images

    def _print_loss(self, loss: torch.Tensor, step: int) -> None:
        if self._ema_loss is None:
            self._ema_loss = loss.item()
        else:
            self._ema_loss = (
                self._ema_const * self._ema_loss
                + (1 - self._ema_const) * loss.item()
            )

        if step % 10 == 0 and self._verbose:
            print(
                f"step: {step:4d}  loss: {self._ema_loss:.4f}  "
                f"time: {time.time() - self._start_time:.2f}s"
            )
            self._start_time = time.time()
