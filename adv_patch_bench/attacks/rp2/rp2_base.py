"""Base class of RP2 Attack."""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from adv_patch_bench.attacks import base_attack
from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import MaskTensor, ImageTensor, Target

# from yolov5.utils.general import non_max_suppression
# from yolov5.utils.plots import output_to_target, plot_images


class RP2AttackModule(base_attack.DetectorAttackModule):
    """Base class of RP2 Attack."""

    def __init__(
        self,
        attack_config: Dict[str, Any],
        core_model: torch.nn.Module,
        **kwargs,
    ):
        """RP2AttackModule is a base class for RP2 Attack.

        Reference: Eykholt et al., "Robust Physical-World Attacks on Deep
        Learning Models," 2018. (https://arxiv.org/abs/1707.08945)

        Args:
            attack_config: Config dict for attacks.
            core_model: Target model to attack.
        """
        super().__init__(attack_config, core_model, **kwargs)
        self.num_steps = attack_config["num_steps"]
        self.step_size = attack_config["step_size"]
        self.optimizer = attack_config["optimizer"]
        self.use_lr_schedule = attack_config["use_lr_schedule"]
        self.num_eot = attack_config["num_eot"]
        self.lmbda = attack_config["lambda"]
        self.min_conf = attack_config["min_conf"]
        self.attack_mode = attack_config["attack_mode"].split("-")
        self.num_restarts = 1
        self.is_training: Optional[bool] = None  # Holding model training state
        self.ema_const = 0.9

        # TODO(feature): Allow more threat models
        self.targeted: bool = False

        # Use change of variable on delta with alpha and beta.
        # Mostly used with per-sign or real attack.
        self.use_var_change_ab = "var_change_ab" in self.attack_mode
        if self.use_var_change_ab:
            # Does not work when num_eot > 1
            assert (
                self.num_eot == 1
            ), "When use_var_change_ab is used, num_eot can only be set to 1."

    @abstractmethod
    def _loss_func(
        self,
        adv_img: ImageTensor,
        adv_target: Target,
        obj_class: int,
    ) -> torch.Tensor:
        """Implement loss function on perturbed image.

        Args:
            adv_img: Image to compute loss on.
            adv_target: Target label to compute loss on.
            obj_class: Target object class. Usually ground-truth label for
                untargeted attack, and target class for targeted attack.

        Returns:
            Loss for attacker to minimize.
        """
        raise NotImplementedError("self._loss_func() not implemented!")

    def compute_loss(
        self,
        delta: ImageTensor,
        adv_img: ImageTensor,
        adv_target: Target,
        obj_class: List[int],
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
        loss: torch.Tensor = self._loss_func(adv_img, adv_target, obj_class)
        tv: torch.Tensor = (delta[:, :-1, :] - delta[:, 1:, :]).abs().mean() + (
            delta[:, :, :-1] - delta[:, :, 1:]
        ).abs().mean()
        loss += self.lmbda * tv
        return loss

    @torch.no_grad()
    def run(
        self,
        rimgs: List[render_image.RenderImage],
        patch_mask: MaskTensor,
    ) -> ImageTensor:
        """Run RP2 Attack.

        Args:

        Returns:
            torch.Tensor: Adversarial patch with shape [C, H, W]
        """
        self._on_enter_attack()
        device = patch_mask.device

        all_bg_idx: np.ndarray = np.arange(len(rimgs))
        # Load patch_mask to all RenderObject first. This should be done once.
        for rimg in rimgs:
            robj = rimg.get_object()
            robj.load_adv_patch(patch_mask=patch_mask)

        for _ in range(self.num_restarts):
            # Initialize adversarial perturbation
            z_delta: ImageTensor = torch.zeros(
                (3,) + patch_mask.shape[-2:],
                device=device,
                dtype=torch.float32,
            )
            z_delta.uniform_(0, 1)

            # Set up optimizer
            opt, lr_schedule = self._setup_opt(z_delta)
            self.ema_loss = None
            self.start_time = time.time()

            # Run PGD on inputs for specified number of steps
            for step in range(self.num_steps):

                # Randomly select RenderImages to attack this step
                np.random.shuffle(all_bg_idx)
                bg_idx = all_bg_idx[: self.num_eot]
                rimg_eot = [rimgs[i] for i in bg_idx]

                # metadata = self._on_real_attack_step(metadata)
                with torch.enable_grad():
                    z_delta.requires_grad_()
                    # Determine how perturbation is projected
                    if self.use_var_change_ab:
                        # TODO(feature): Does not work when num_eot > 1
                        robj = rimg_eot[0].get_object()
                        alpha, beta = robj.alpha, robj.beta
                        delta = self._to_model_space(
                            z_delta, beta, alpha + beta
                        )
                    else:
                        delta = self._to_model_space(z_delta, 0, 1)

                    # Load new adversarial patch to each RenderObject
                    obj_class: int = robj.obj_class
                    # TODO(feature): Support batch rimg
                    rimg = rimg_eot[0]
                    robj = rimg.get_object()
                    robj.load_adv_patch(adv_patch=delta)
                    # Apply adversarial patch to each RenderImage
                    adv_img, adv_target = rimg.apply_objects()
                    adv_img = rimg.post_process_image(adv_img)

                    # DEBUG
                    # if step % 100 == 0:
                    #     torchvision.utils.save_image(
                    #         adv_img[0], f'gen_adv_real_{step}.png')

                    # mdata = None if metadata is None else metadata[bg_idx]
                    loss: torch.Tensor = self.compute_loss(
                        delta, adv_img, adv_target, obj_class
                    )
                    loss.backward()
                    z_delta = self._step_opt(z_delta, opt)

                if lr_schedule is not None:
                    lr_schedule.step(loss)
                self._print_loss(loss, step)

                # DEBUG
                # import os
                # for idx in range(self.num_eot):
                #     if not os.path.exists(f'tmp/{idx}/test_adv_img_{step}.png'):
                #         os.makedirs(f'tmp/{idx}/', exist_ok=True)
                #     torchvision.utils.save_image(adv_img[idx], f'tmp/{idx}/test_adv_img_{step}.png')

        # DEBUG
        # outt = non_max_suppression(out.detach(), conf_thres=0.25, iou_thres=0.45)
        # plot_images(adv_img.detach(), output_to_target(outt))

        self._on_exit_attack()
        # Return worst-case perturbed input logits
        return delta.detach()

    def _setup_opt(
        self, z_delta: ImageTensor
    ) -> Tuple[Optional[optim.Optimizer], Optional[Any]]:
        # Set up optimizer
        if self.optimizer == "sgd":
            opt = optim.SGD([z_delta], lr=self.step_size, momentum=0.999)
        elif self.optimizer == "adam":
            opt = optim.Adam([z_delta], lr=self.step_size)
        elif self.optimizer == "rmsprop":
            opt = optim.RMSprop([z_delta], lr=self.step_size)
        elif self.optimizer == "pgd":
            opt = None
        else:
            raise NotImplementedError("Given optimizer not implemented.")

        lr_schedule = None
        if self.use_lr_schedule and opt is not None:
            lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=0.5,
                patience=int(self.num_steps / 10),
                threshold=1e-9,
                min_lr=self.step_size * 1e-6,
                verbose=self._verbose,
            )

        return opt, lr_schedule

    def _step_opt(
        self, z_delta: ImageTensor, opt: Optional[optim.Optimizer]
    ) -> ImageTensor:
        if self.optimizer == "pgd":
            grad = z_delta.grad.detach()
            grad = torch.sign(grad)
            z_delta = z_delta.detach() - self.step_size * grad
            z_delta.clamp_(0, 1)
        else:
            opt.step()
        return z_delta

    def _to_model_space(self, x, min_, max_):
        """Transforms an input from the attack space to the model space."""
        if "pgd" in self.attack_mode:
            return x

        # from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)

        # map from (-1, +1) to (min_, max_)
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a
        return x

    def _print_loss(self, loss: torch.Tensor, step: int) -> None:
        if self.ema_loss is None:
            self.ema_loss = loss.item()
        else:
            self.ema_loss = (
                self.ema_const * self.ema_loss
                + (1 - self.ema_const) * loss.item()
            )

        if step % 100 == 0 and self._verbose:
            print(
                f"step: {step:4d}  loss: {self.ema_loss:.4f}  "
                f"time: {time.time() - self.start_time:.2f}s"
            )
            self.start_time = time.time()
