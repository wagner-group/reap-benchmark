"""Implement REAP patch rendering for each object."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import kornia.geometry.transform as kornia_tf
import numpy as np
import torch

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms import render_object
from adv_patch_bench.transforms.util import identity
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    BatchImageTensorRGBA,
    BatchMaskTensor,
    ImageTensor,
    Target,
)

_VALID_TRANSFORM_MODE = ("perspective", "translate_scale")
_EPS = 1e-6

logger = logging.getLogger(__name__)


class ReapObject(render_object.RenderObject):
    """Object wrapper using REAP benchmark."""

    def __init__(
        self,
        obj_dict: dict[str, Any] | None = None,
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
        self._patch_transform_mode: str = patch_transform_mode
        self._use_patch_relight: bool = use_patch_relight

        # # Get REAP relighting transform params
        if use_patch_relight:
            alpha = torch.tensor(
                obj_dict["alpha"], device=self._device, dtype=torch.float32
            )
            beta = torch.tensor(
                obj_dict["beta"], device=self._device, dtype=torch.float32
            )
        else:
            alpha = torch.tensor(1.0, device=self._device, dtype=torch.float32)
            beta = torch.tensor(0.0, device=self._device, dtype=torch.float32)
        self.alpha: torch.Tensor = img_util.coerce_rank(alpha, 4)
        self.beta: torch.Tensor = img_util.coerce_rank(beta, 4)

        # Get REAP geometric transform params
        self.transform_mat = self._get_reap_transforms(obj_dict["keypoints"])

    def _get_reap_transforms(
        self, tgt: np.ndarray | list[list[float]] | None = None
    ) -> torch.Tensor:
        """Get transformation matrix and parameters.

        Returns:
            Tuple of (Transform function, transformation matrix, target points).
        """
        tgt = np.array(tgt, dtype=np.float32)[:, :2]
        src: np.ndarray = self.src_points
        tgt = tgt[: len(src)].copy()

        if self._patch_transform_mode == "translate_scale":
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

        assert src.shape == tgt.shape, (
            f"src and tgt keypoints don't have the same shape ({src.shape} vs "
            f"{tgt.shape})!"
        )

        if len(src) == 3:
            # For triangles which have only 3 keypoints
            transform_mat = cv2.getAffineTransform(src, tgt)
            transform_mat = torch.from_numpy(transform_mat).unsqueeze(0).float()
            new_row = torch.tensor([[[0, 0, 1]]])
            transform_mat = torch.cat([transform_mat, new_row], dim=1)
        else:
            # All other signs use perspective transform
            src = torch.from_numpy(src).unsqueeze(0)
            tgt = torch.from_numpy(tgt).unsqueeze(0)
            transform_mat = kornia_tf.get_perspective_transform(src, tgt)
        return transform_mat.to(self._device)

    @staticmethod
    def apply_objects(
        images: BatchImageTensor,
        targets: list[Target],
        adv_patch: BatchImageTensor,
        patch_mask: BatchMaskTensor,
        tf_params: dict[str, Any],
        suppress_aug: bool = False,
    ) -> tuple[ImageTensor, Target]:
        """Apply adversarial patch to image using REAP approach.

        Args:
            image: Image to apply patch to.
            target: Target labels (unmodified).

        Returns:
            final_img: Image with transformed patch applied.
            target: Target with synthetic object label added.
        """
        transform_mat = tf_params["transform_mat"]
        alpha = tf_params["alpha"]
        beta = tf_params["beta"]
        obj_to_img = tf_params["obj_to_img"]
        obj_mask = tf_params["obj_mask"]
        aug_geo, aug_light = identity, identity
        if not suppress_aug:
            aug_geo, _, aug_light = tf_params["obj_transforms"]

        if adv_patch.shape[-2:] != patch_mask.shape[-2:] != obj_mask.shape[-2:]:
            raise ValueError(
                f"Shape mismatched: adv_patch {adv_patch.shape}, patch_mask "
                f"{patch_mask.shape}, obj_mask: {obj_mask.shape}!"
            )
        batch_size: int = len(images)
        num_objs: int = len(obj_to_img)
        if len(targets) != batch_size:
            raise IndexError(
                f"targets and images must have the same length ({len(targets)} "
                f"vs {batch_size})!"
            )
        if any(len(inpt) != num_objs for inpt in (transform_mat, alpha, beta)):
            raise IndexError(
                "Transform data must have length equal to the number of objects"
                f" ({num_objs})!"
            )
        for inpt in (adv_patch, patch_mask, obj_mask):
            if len(inpt) not in (num_objs, 1):
                raise IndexError(
                    "Patch and masks must have the same length as the number "
                    f"of objects but see {len(inpt)} vs {num_objs}!"
                )

        # Apply relighting transform (brightness and contrast)
        if adv_patch.is_leaf:
            adv_patch = adv_patch * alpha
        else:
            adv_patch.mul_(alpha)
        adv_patch.add_(beta)
        adv_patch.clamp_(0 + _EPS, 1 - _EPS)

        # Apply extra lighting augmentation on patch
        adv_patch = aug_light(adv_patch)

        # Combine patch_mask with adv_patch as alpha channel
        rgba_patch: BatchImageTensorRGBA = torch.cat(
            [adv_patch, patch_mask], dim=1
        )
        # Crop with patch_mask
        rgba_patch *= patch_mask

        # Apply extra geometric augmentation on patch
        rgba_patch = aug_geo(rgba_patch)
        rgba_patch *= obj_mask

        # Apply transform on RGBA patch
        warped_patch: BatchImageTensorRGBA = kornia_tf.warp_perspective(
            rgba_patch,
            transform_mat,
            images.shape[-2:],
            mode=tf_params["interp"],
            padding_mode="zeros",
        )
        warped_patch.clamp_(0, 1)

        # Add patches from same image together
        # per_img_patches = warped_patch
        # if obj_to_img is not None:
        per_img_patches = []
        for i in range(batch_size):
            per_img_patches.append(
                warped_patch[obj_to_img == i].sum(0, keepdim=True)
            )
        per_img_patches = torch.cat(per_img_patches, dim=0)
        assert len(per_img_patches) == batch_size

        alpha_mask: BatchImageTensor = per_img_patches[:, -1:]
        warped_patch: BatchImageTensor = per_img_patches[:, :-1]
        num_overlap_pixels = (alpha_mask > 1).sum().item()
        if num_overlap_pixels > 0:
            # This only happens when two or more patches overlap. Ideally we
            # should select one patch to be on top covering the others, but
            # this is usually rare enough that we can fix by just clipping.
            alpha_mask.clamp_max_(1)
            warped_patch.clamp_max_(1)
            logger.debug(
                "  %d pixels overlap! If this number is large, geometric "
                "transformation is likely wrong.",
                num_overlap_pixels,
            )
            logger.debug(str([t["file_name"].split("/")[-1] for t in targets]))
            # print(transform_mat)
            # img = (1 - alpha_mask) * images + alpha_mask * warped_patch
            # transform_mat[0] @ torch.tensor([0, 0, 1], device="cuda", dtype=torch.float32)
            # import torchvision
            # torchvision.utils.save_image(img, "temp.png")
            # torchvision.utils.save_image(alpha_mask, "temp_alpha.png")
            # import pdb
            # pdb.set_trace()

        # Place patch on object using alpha channel
        final_img: BatchImageTensor = (
            1 - alpha_mask
        ) * images + alpha_mask * warped_patch

        # DEPRECATED: This is checked by render image
        # if final_img.isnan().any():
        #     logger.warning(
        #         "NaN value(s) found in REAP rendered images! Returning originals..."
        #     )
        #     final_img = images

        return final_img, targets

    def aggregate_params(self, params_dicts: list[dict[str, Any]]) -> None:
        """Append self transform params to params_dicts."""
        params = {
            "transform_mat": self.transform_mat,
            "alpha": self.alpha,
            "beta": self.beta,
            "obj_mask": self.obj_mask,
        }
        for name, value in params.items():
            if name in params_dicts:
                params_dicts[name].append(value)
            else:
                params_dicts[name] = [value]
