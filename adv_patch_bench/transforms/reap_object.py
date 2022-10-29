"""Implement REAP patch rendering for each object."""

from typing import Any, Callable, Tuple

import adv_patch_bench.utils.image as img_util
import cv2
import kornia.geometry.transform as kornia_tf
import numpy as np
import pandas as pd
import torch
from adv_patch_bench.transforms import render_object
from adv_patch_bench.utils.types import (
    BatchImageTensorGeneric,
    BatchImageTensorRGBA,
    ImageTensor,
    ImageTensorRGBA,
    Target,
    TransformFn,
)

_VALID_TRANSFORM_MODE = ("perspective", "translate_scale")


class ReapObject(render_object.RenderObject):
    """Object wrapper using REAP benchmark."""

    def __init__(
        self,
        patch_transform_mode: str = "perspective",
        use_patch_relight: bool = True,
        **kwargs,
    ) -> None:
        """Initialize ReapObject.

        Args:
            patch_transform_mode: Type of geometric transform functions to use.
                Defaults to "perspective".
            use_patch_relight: Whether to apply relighting transform to
                adversarial patch. Defaults to True.

        Raises:
            NotImplementedError: Invalid transform mode.
        """
        super().__init__(**kwargs)

        if patch_transform_mode not in _VALID_TRANSFORM_MODE:
            raise NotImplementedError(
                f"transform_mode {patch_transform_mode} is not implemented. "
                f"Only supports {_VALID_TRANSFORM_MODE}!"
            )
        self.patch_transform_mode: str = patch_transform_mode

        # Get REAP relighting transform params
        if use_patch_relight:
            alpha = torch.tensor(self.obj_df["alpha"], device=self._device)
            beta = torch.tensor(self.obj_df["beta"], device=self._device)
        else:
            alpha = torch.tensor(1.0, device=self._device)
            beta = torch.tensor(0.0, device=self._device)
        self.alpha: torch.Tensor = img_util.coerce_rank(alpha, 3)
        self.beta: torch.Tensor = img_util.coerce_rank(beta, 3)

        # Get REAP geometric transform params
        tf_data = self._get_reap_transforms(self.obj_df)
        self.transform_mat = tf_data[1].to(self._device)
        self.transform_fn: TransformFn = self._wrap_transform_fn(tf_data[0])

    def _wrap_transform_fn(
        self, transform_fn: Callable[..., BatchImageTensorGeneric]
    ) -> TransformFn:
        """Wrap kornia transform function to avoid passing arguments around.

        Args:
            transform_fn: kornia transform function to wrap.

        Returns:
            Wrapped transform function that only takes image as input.
        """

        def wrapper_tf_fn(
            x: BatchImageTensorGeneric,
        ) -> BatchImageTensorGeneric:
            return transform_fn(
                x,
                self.transform_mat,
                self.img_size,
                mode=self._interp,
                padding_mode="zeros",
            )

        return wrapper_tf_fn

    def _get_reap_transforms(
        self, df_row: pd.DataFrame
    ) -> Tuple[Callable[..., Any], torch.Tensor]:
        """Get transformation matrix and parameters.

        Returns:
            Tuple of (Transform function, transformation matrix, target points).
        """
        h_ratio, w_ratio = self.img_hw_ratio
        h_pad, w_pad = self.img_pad_size

        tgt = np.array(df_row["tgt_points"], dtype=np.float32)
        tgt[:, 1] = (tgt[:, 1] * h_ratio) + h_pad
        tgt[:, 0] = (tgt[:, 0] * w_ratio) + w_pad
        src = np.array(self.src_points, dtype=np.float32)

        if self.patch_transform_mode == "translate_scale":
            # Use corners of axis-aligned bounding box for transform
            # (translation and scaling) instead of real corners.
            min_tgt_x = min(tgt[:, 0])
            max_tgt_x = max(tgt[:, 0])
            min_tgt_y = min(tgt[:, 1])
            max_tgt_y = max(tgt[:, 1])
            tgt = np.array(
                [
                    [min_tgt_x, min_tgt_y],
                    [max_tgt_x, min_tgt_y],
                    [max_tgt_x, max_tgt_y],
                    [min_tgt_x, max_tgt_y],
                ]
            )

            min_src_x = min(src[:, 0])
            max_src_x = max(src[:, 0])
            min_src_y = min(src[:, 1])
            max_src_y = max(src[:, 1])
            src = np.array(
                [
                    [min_src_x, min_src_y],
                    [max_src_x, min_src_y],
                    [max_src_x, max_src_y],
                    [min_src_x, max_src_y],
                ]
            )

        # Get transformation matrix and transform function from source and
        # target keypoint coordinates.
        if len(src) == 3:
            # For triangles which have only 3 keypoints
            M = (
                torch.from_numpy(cv2.getAffineTransform(src, tgt))
                .unsqueeze(0)
                .float()
            )
            transform_func = kornia_tf.warp_affine
        else:
            # All other signs use perspective transform
            src = torch.from_numpy(src).unsqueeze(0)
            tgt = torch.from_numpy(tgt).unsqueeze(0)
            M = kornia_tf.get_perspective_transform(src, tgt)
            transform_func = kornia_tf.warp_perspective

        return transform_func, M

    def apply_object(
        self,
        image: ImageTensor,
        target: Target,
    ) -> Tuple[ImageTensor, Target]:
        """Apply adversarial patch to image using REAP approach.

        Args:
            image: Image to apply patch to.
            target: Target labels (unmodified).

        Returns:
            final_img: Image with transformed patch applied.
            target: Target with synthetic object label added.
        """
        adv_patch: ImageTensor = self.adv_patch.clone()
        patch_mask: ImageTensor = self.patch_mask.clone()

        adv_patch = img_util.coerce_rank(adv_patch, 3)
        patch_mask = img_util.coerce_rank(patch_mask, 3)
        if not (
            adv_patch.shape[-2:] == patch_mask.shape[-2:] == self.obj_size_px
        ):
            raise ValueError(
                f"Shape mismatched: adv_patch {adv_patch.shape}, patch_mask "
                f"{patch_mask.shape}, obj_size {self.obj_size_px}!"
            )

        # Apply relighting transform (brightness and contrast)
        adv_patch.mul_(self.alpha).add_(self.beta)
        adv_patch.clamp_(0, 1)

        # Apply extra lighting augmentation on patch
        adv_patch = self.aug_light(adv_patch)
        adv_patch.clamp_(0, 1)

        # Combine patch_mask with adv_patch as alpha channel
        alpha_patch: ImageTensorRGBA = torch.cat([adv_patch, patch_mask], dim=0)
        # Crop with sign_mask and patch_mask
        alpha_patch *= self.obj_mask * patch_mask

        # Apply extra geometric augmentation on patch
        alpha_patch: BatchImageTensorRGBA
        alpha_patch, _ = self.aug_geo(alpha_patch)
        alpha_patch = img_util.coerce_rank(alpha_patch, 4)

        # Apply transform on RGBA patch
        warped_patch: BatchImageTensorRGBA = self.transform_fn(alpha_patch)
        warped_patch.squeeze_(0)
        warped_patch.clamp_(0, 1)

        # Place patch on object using alpha channel
        alpha_mask = warped_patch[-1:]
        warped_patch: ImageTensor = warped_patch[:-1]
        final_img: ImageTensor = (
            1 - alpha_mask
        ) * image + alpha_mask * warped_patch

        return final_img, target
