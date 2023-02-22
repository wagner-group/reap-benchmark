"""Define utility functions for creating patch masks."""

from __future__ import annotations

import torch

from adv_patch_bench.utils.types import MaskTensor, SizeMM, SizePx


def _inch_to_mm(length_in_inch: int | float) -> float:
    return 25.4 * length_in_inch


def _gen_mask_rect(
    patch_size_mm: tuple[int, float, float],
    obj_size_px: SizePx,
    obj_size_mm: SizeMM,
    patch_loc: str | float | None = None,
) -> MaskTensor:
    """Generate rectangular patch mask at the bottom of the object.

    If num_patches is 2, the second patch is placed at the top of the object.

    Args:
        patch_size_mm: Patch size in millimeters.
        obj_size_px: Object size in pixels.
        obj_size_mm: Object size in millimeters.
        patch_loc: Height to shift patch from the bottom edge of the sign.
            Defaults to 0.

    Returns:
        Binary mask of patch.
    """
    if isinstance(patch_loc, (int, float)) and patch_loc < 0:
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
    if isinstance(patch_loc, (int, float)):
        shift_mm = patch_loc  # How much to shift down from middle
    elif patch_loc == "middle":
        mid_height = 0
        shift_mm = obj_h_mm / 2
    elif patch_loc == "top":
        shift_mm = -(obj_h_mm - patch_h_mm) / 2
    else:
        # Bottom (default)
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
    patch_size: str,
    obj_size_px: SizePx,
    obj_size_mm: SizeMM,
) -> MaskTensor:
    """Generate digital patch mask with given real patch_size_mm.

    Args:
        patch_size: String describing patch size in format of
            <NUM_PATCHES>_<HEIGHT>x<WIDTH>_<LOCATION>.
        obj_size_px: Size of object to place patch on in pixels.
        obj_size_mm: Size of object to place patch on in millimeters.

    Raises:
        ValueError: Invalid format for patch_size.

    Returns:
        Patch mask (rank 3 and first rank has dimension of 1).
    """
    px_ratio = obj_size_px[0] / obj_size_px[1]
    mm_ratio = obj_size_mm[0] / obj_size_mm[1]
    if abs(px_ratio - mm_ratio) > 1e-2:
        raise ValueError(
            "Aspect ratio of obj_size_px and obj_size_mm must match "
            f"({px_ratio} vs {mm_ratio})!"
        )

    # patch_size has format <NUM_PATCHES>_<HEIGHT>x<WIDTH>_<LOCATION>
    patch_tokens = patch_size.split("_")
    if len(patch_tokens) != 3:
        raise ValueError(
            f"Invalid patch size. Must use the following format: "
            f"<NUM_PATCHES>_<HEIGHT>x<WIDTH>_<LOCATION> but got {patch_size}!"
        )
    num_patches: int = int(patch_tokens[0])
    if num_patches not in (1, 2):
        raise NotImplementedError(
            f"Only num_patches of 1 or 2 for now, but {num_patches} is given "
            f"({patch_size})!"
        )

    patch_size = patch_tokens[1].split("x")
    if not all(s.isnumeric() for s in patch_size):
        raise ValueError(f"Invalid patch size: {patch_size}!")
    patch_size_inch = [int(s) for s in patch_size]
    patch_size_mm = [_inch_to_mm(s) for s in patch_size_inch]
    patch_size_mm = (num_patches,) + tuple(patch_size_mm)

    # TODO(feature): Add other non-rect patch shape
    patch_mask: MaskTensor = _gen_mask_rect(
        patch_size_mm,
        obj_size_px,
        obj_size_mm,
        patch_loc=patch_tokens[2],
    )

    return patch_mask
