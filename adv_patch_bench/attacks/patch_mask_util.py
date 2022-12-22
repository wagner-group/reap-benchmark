"""Define utility functions for creating patch masks."""

from __future__ import annotations

import torch

from adv_patch_bench.utils.types import MaskTensor, SizeMM, SizePx


def _gen_mask_rect(
    patch_size_mm: tuple[int, float, float],
    obj_size_px: SizePx,
    obj_size_mm: SizeMM,
    patch_height: str | float | None = None,
) -> MaskTensor:
    """Generate rectangular patch mask at the bottom of the object.

    If num_patches is 2, the second patch is placed at the top of the object.

    Args:
        patch_size_mm: Patch size in millimeters.
        obj_size_px: Object size in pixels.
        obj_size_mm: Object size in millimeters.
        shift_height_mm: Height to shift patch from the bottom edge of the sign.
            Defaults to 0.

    Returns:
        Binary mask of patch.
    """
    if isinstance(patch_height, (int, float)) and patch_height < 0:
        raise ValueError("shift_height_mm must be non-negative!")
    patch_mask: MaskTensor = torch.zeros(
        (1,) + obj_size_px, dtype=torch.float32
    )  # type: ignore
    obj_h_px, obj_w_px = obj_size_px
    obj_h_mm, obj_w_mm = obj_size_mm
    num_patches, patch_h_mm, patch_w_mm = patch_size_mm
    patch_h_px = round(patch_h_mm / obj_h_mm * obj_h_px)
    patch_w_px = round(patch_w_mm / obj_w_mm * obj_w_px)

    # Define patch location and size
    mid_height, mid_width = obj_h_px // 2, obj_w_px // 2
    if isinstance(patch_height, (int , float)):
        shift_mm = patch_height
    elif patch_height == "middle":
        mid_height = 0
        shift_mm = obj_h_mm / 2
    else:
        shift_mm = (obj_h_mm - patch_h_mm) / 2
    patch_y_shift = round(shift_mm / obj_h_mm * obj_h_px)
    patch_x_pos = mid_width
    hh, hw = patch_h_px // 2, patch_w_px // 2

    # Bottom patch
    patch_y_pos = mid_height + patch_y_shift
    patch_mask[
        :,
        max(0, patch_y_pos - hh) : patch_y_pos + hh,
        max(0, patch_x_pos - hw) : patch_x_pos + hw,
    ] = 1

    if num_patches == 2:
        # Top patch
        patch_y_pos = mid_height - patch_y_shift
        patch_mask[
            :,
            max(0, patch_y_pos - hh) : max(0, patch_y_pos + hh),
            max(0, patch_x_pos - hw) : patch_x_pos + hw,
        ] = 1

    return patch_mask


def gen_patch_mask(
    patch_size_mm: tuple[int, float, float],
    obj_size_px: SizePx,
    obj_size_mm: SizeMM,
    patch_height: str | float | None = None,
) -> MaskTensor:
    """Generate digital patch mask with given real patch_size_mm.

    Args:
        patch_size_mm: Tuple (num_patch, height, width). Height and width are
            in millimeters, and num_patch must be int.
        obj_size_px: Size of object to place patch on in pixels.
        obj_size_mm: Size of object to place patch on in millimeters.
        shift_height_mm: Height to shift patch from the bottom edge of the sign.
            Defaults to 0.

    Raises:
        ValueError: Invalid format for patch_size_mm.

    Returns:
        Patch mask (rank 3 and first rank has dimension of 1).
    """
    if not isinstance(patch_size_mm, tuple) or len(patch_size_mm) != 3:
        raise ValueError(
            "patch_size_mm must be a tuple of three numbers (num_patch, "
            f"height, width), but {patch_size_mm} is given!"
        )

    px_ratio = obj_size_px[0] / obj_size_px[1]
    mm_ratio = obj_size_mm[0] / obj_size_mm[1]
    if abs(px_ratio - mm_ratio) > 1e-2:
        raise ValueError(
            "Aspect ratio of obj_size_px and obj_size_mm must match "
            f"({px_ratio} vs {mm_ratio})!"
        )

    # TODO(feature): Add other non-rect patch shape
    patch_mask: MaskTensor = _gen_mask_rect(
        patch_size_mm,
        obj_size_px,
        obj_size_mm,
        patch_height=patch_height,
    )

    return patch_mask
