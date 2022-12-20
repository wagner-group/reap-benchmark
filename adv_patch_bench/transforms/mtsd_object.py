"""Implement REAP patch rendering for each object."""

from __future__ import annotations

# from typing import Any

# import cv2
# import kornia.geometry.transform as kornia_tf
# import numpy as np
# import torch

from adv_patch_bench.transforms import reap_object
# from adv_patch_bench.utils.types import (
#     BatchImageTensor,
#     BatchImageTensorRGBA,
#     ImageTensor,
#     ImageTensorRGBA,
#     Target,
# )


class MtsdObject(reap_object.ReapObject):
    """Object wrapper for MTSD samples."""

    def __init__(self, **kwargs) -> None:
        """Initialize MtsdObject."""
        super().__init__(use_box_mode=True, **kwargs)

    # def _get_reap_transforms(
    #     self, tgt: np.ndarray | list[list[float]] | None = None
    # ) -> torch.Tensor:
    #     """Get transformation matrix and parameters.

    #     Returns:
    #         Tuple of (Transform function, transformation matrix, target points).
    #     """
    #     tgt = np.array(tgt, dtype=np.float32)[:, :2]
    #     src = np.array(self.src_points, dtype=np.float32)
    #     tgt = tgt[: len(src)].copy()

    #     if self._patch_transform_mode == "translate_scale":
    #         # Use corners of axis-aligned bounding box for transform
    #         # (translation and scaling) instead of real corners.
    #         min_tgt_x = min(tgt[:, 0])
    #         max_tgt_x = max(tgt[:, 0])
    #         min_tgt_y = min(tgt[:, 1])
    #         max_tgt_y = max(tgt[:, 1])
    #         tgt = np.array(
    #             [
    #                 [min_tgt_x, min_tgt_y],
    #                 [max_tgt_x, min_tgt_y],
    #                 [max_tgt_x, max_tgt_y],
    #                 [min_tgt_x, max_tgt_y],
    #             ]
    #         )

    #         min_src_x = min(src[:, 0])
    #         max_src_x = max(src[:, 0])
    #         min_src_y = min(src[:, 1])
    #         max_src_y = max(src[:, 1])
    #         src = np.array(
    #             [
    #                 [min_src_x, min_src_y],
    #                 [max_src_x, min_src_y],
    #                 [max_src_x, max_src_y],
    #                 [min_src_x, max_src_y],
    #             ]
    #         )

    #     assert src.shape == tgt.shape, (
    #         f"src and tgt keypoints don't have the same shape ({src.shape} vs "
    #         f"{tgt.shape})!"
    #     )

    #     if len(src) == 3:
    #         # For triangles which have only 3 keypoints
    #         transform_mat = cv2.getAffineTransform(src, tgt)
    #         transform_mat = torch.from_numpy(transform_mat).unsqueeze(0).float()
    #         new_row = torch.tensor([[[0, 0, 1]]])
    #         transform_mat = torch.cat([transform_mat, new_row], dim=1)
    #     else:
    #         # All other signs use perspective transform
    #         src = torch.from_numpy(src).unsqueeze(0)
    #         tgt = torch.from_numpy(tgt).unsqueeze(0)
    #         transform_mat = kornia_tf.get_perspective_transform(src, tgt)
    #     return transform_mat.to(self._device)

    # @staticmethod
    # def apply_objects(
    #     images: BatchImageTensor,
    #     targets: Target,
    #     adv_patch,
    #     patch_mask,
    #     tf_params: dict[str, Any],
    # ) -> tuple[ImageTensor, Target]:
    #     """Apply adversarial patch to image using REAP approach.

    #     Args:
    #         image: Image to apply patch to.
    #         target: Target labels (unmodified).

    #     Returns:
    #         final_img: Image with transformed patch applied.
    #         target: Target with synthetic object label added.
    #     """
    #     transform_mat = tf_params["transform_mat"]
    #     alpha = tf_params["alpha"]
    #     beta = tf_params["beta"]
    #     obj_to_img = tf_params["obj_to_img"]
    #     obj_mask = tf_params["obj_mask"]
    #     aug_geo, _, aug_light = tf_params["obj_transforms"]

    #     if adv_patch.shape[-2:] != patch_mask.shape[-2:] != obj_mask.shape[-2:]:
    #         raise ValueError(
    #             f"Shape mismatched: adv_patch {adv_patch.shape}, patch_mask "
    #             f"{patch_mask.shape}, obj_mask: {obj_mask.shape}!"
    #         )
    #     batch_size: int = len(images)
    #     num_objs: int = len(obj_to_img)
    #     if len(targets) != batch_size:
    #         raise IndexError(
    #             f"targets and images must have the same length ({len(targets)} "
    #             f"vs {batch_size})!"
    #         )
    #     if any(len(inpt) != num_objs for inpt in (transform_mat, alpha, beta)):
    #         raise IndexError(
    #             "Transform data must have length equal to the number of objects"
    #             f" ({num_objs})!"
    #         )
    #     for inpt in (adv_patch, patch_mask, obj_mask):
    #         if len(inpt) not in (num_objs, 1):
    #             raise IndexError("Patch and masks must have ")

    #     # Apply relighting transform (brightness and contrast)
    #     adv_patch.mul_(alpha).add_(beta)
    #     adv_patch.clamp_(0, 1)

    #     # Apply extra lighting augmentation on patch
    #     adv_patch = aug_light(adv_patch)
    #     adv_patch.clamp_(0, 1)

    #     # Combine patch_mask with adv_patch as alpha channel
    #     rgba_patch: ImageTensorRGBA = torch.cat([adv_patch, patch_mask], dim=1)
    #     # Crop with sign_mask and patch_mask
    #     rgba_patch *= obj_mask * patch_mask

    #     # Apply extra geometric augmentation on patch
    #     rgba_patch: BatchImageTensorRGBA
    #     rgba_patch, _ = aug_geo(rgba_patch)

    #     # Apply transform on RGBA patch
    #     warped_patch: BatchImageTensorRGBA = kornia_tf.warp_perspective(
    #         rgba_patch,
    #         transform_mat,
    #         images.shape[-2:],
    #         mode=tf_params["interp"],
    #         padding_mode="zeros",
    #     )
    #     warped_patch.clamp_(0, 1)

    #     # Add patches from same image together
    #     per_img_patches: list[BatchImageTensorRGBA] = []
    #     for i in range(batch_size):
    #         per_img_patches.append(
    #             warped_patch[obj_to_img == i].sum(0, keepdim=True)
    #         )
    #     per_img_patches = torch.cat(per_img_patches, dim=0)
    #     assert len(per_img_patches) == batch_size

    #     # Place patch on object using alpha channel
    #     alpha_mask = per_img_patches[:, -1:]
    #     warped_patch: BatchImageTensor = per_img_patches[:, :-1]
    #     final_img: ImageTensor = (
    #         1 - alpha_mask
    #     ) * images + alpha_mask * warped_patch

    #     return final_img, targets

    # def aggregate_params(self, params_dict):
    #     params = {
    #         "transform_mat": self.transform_mat,
    #         "alpha": self.alpha,
    #         "beta": self.beta,
    #         "obj_mask": self.obj_mask,
    #     }
    #     for name, value in params.items():
    #         if name in params_dict:
    #             params_dict[name].append(value)
    #         else:
    #             params_dict[name] = [value]
