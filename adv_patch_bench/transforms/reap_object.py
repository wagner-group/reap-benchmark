"""Implement REAP patch rendering for each object."""

from __future__ import annotations

import logging
from typing import Any

import kornia.geometry.transform as kornia_tf
import torch

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms import render_object
from adv_patch_bench.transforms.geometric_tf import get_transform_matrix
from adv_patch_bench.transforms.lighting_tf import RelightTransform
from adv_patch_bench.transforms.util import identity
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    BatchImageTensorRGBA,
    BatchMaskTensor,
    ImageTensor,
    Target,
)

_EPS = 1e-6
logger = logging.getLogger(__name__)


class ReapObject(render_object.RenderObject):
    """Object wrapper using REAP benchmark."""

    def __init__(
        self,
        obj_dict: dict[str, Any] | None = None,
        reap_geo_method: str = "perspective",
        reap_relight_method: str = "color_transfer",
        **kwargs,
    ) -> None:
        """Initialize ReapObject.

        Args:
            obj_dict: Dictionary containing object parameters.
            reap_geo_method: Type of geometric transform functions to use.
                Defaults to "perspective".
            reap_relight_method: Type of geometric transform functions to use.
                Defaults to "color_transfer".

        Raises:
            NotImplementedError: Invalid transform mode.
        """
        if "dataset" not in kwargs:
            kwargs["dataset"] = "reap"
        if "pad_to_square" not in kwargs:
            kwargs["pad_to_square"] = True
        if "use_box_mode" not in kwargs:
            kwargs["use_box_mode"] = False
        super().__init__(**kwargs)
        # Get REAP relighting transform params
        relight_coeffs = torch.tensor(
            obj_dict[f"{reap_relight_method}_coeffs"]
            if reap_relight_method != "none"
            else [[1, 0]],
            device=self._device,
            dtype=torch.float32,
        )
        self.relight_coeffs: torch.Tensor = img_util.coerce_rank(
            relight_coeffs, 3
        )
        self.relight_transform = RelightTransform(reap_relight_method).to(
            self._device
        )

        # Get REAP geometric transform params
        self.transform_mat = get_transform_matrix(
            src=self.src_points,
            tgt=obj_dict["keypoints"],
            transform_mode=reap_geo_method,
        ).to(self._device)

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
        if adv_patch is None or patch_mask is None:
            return images, targets

        transform_mat = tf_params["transform_mat"]
        relight_coeffs = tf_params["relight_coeffs"]
        obj_to_img = tf_params["obj_to_img"]
        obj_mask = tf_params["obj_mask"]
        relight_transform = tf_params["relight_transform"]
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
        if any(len(m) != num_objs for m in (transform_mat, relight_coeffs)):
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
        adv_patch = relight_transform(adv_patch, relight_coeffs)
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

        # Place patch on object using alpha channel
        final_img: BatchImageTensor = (
            1 - alpha_mask
        ) * images + alpha_mask * warped_patch

        return final_img, targets

    def aggregate_params(self, params_dicts: list[dict[str, Any]]) -> None:
        """Append self transform params to params_dicts."""
        params = {
            "transform_mat": self.transform_mat,
            "relight_coeffs": self.relight_coeffs,
            "obj_mask": self.obj_mask,
        }
        for name, value in params.items():
            if name in params_dicts:
                params_dicts[name].append(value)
            else:
                params_dicts[name] = [value]
        if "relight_transform" not in params_dicts:
            params_dicts["relight_transform"] = self.relight_transform
