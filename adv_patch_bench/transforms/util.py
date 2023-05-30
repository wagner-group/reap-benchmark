"""Utility functions for transforms."""

from __future__ import annotations

from typing import List, NewType, Tuple

import kornia
import kornia.augmentation as K
import numpy as np
import torch
from torch import nn

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.utils.types import BatchImageTensor, TransformFn

_KeyPoints = NewType("_KeyPoints", List[Tuple[float, float]])


class RandomNoise(nn.Module):
    """Additive uniform noise."""

    def __init__(self, p: float = 1.0, max_range: float = 0.15) -> None:
        """Initialize RandomNoise.

        Args:
            p: Probability of applying this transform. Defaults to 1.0.
            max_range: Maximum range of noise [0, 1]. Noise is in range
                [-max_range, max_range]. Defaults to 0.15.
        """
        super().__init__()
        self.prob: float = p
        self.max_range: float = max_range

    def forward(self, images: BatchImageTensor) -> BatchImageTensor:
        """Add random noise to input tensor."""
        if torch.rand(1) > self.prob:
            return images
        images = images + (torch.rand_like(images) - 0.5) * (2 * self.max_range)
        images.clamp_(0, 1)
        return images


identity = nn.Identity()


def _gen_rect_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate rectangular mask.

    The keypoints are a list of tuple (x, y) coordinates starting from the
    uppper left one and sorted clockwise. For example, keypoints of a
    rectangular mask are upper-left, upper-right, lower-right, and lower-left
    corners, respectively.

    Args:
        size: Width of object in pixels.
        ratio: Ratio between height and width.

    Returns:
        Binary mask and source keypoint for geometric transformation with
        respect to this mask.
    """
    height: int = round(ratio * size)
    width: int = size
    mask: np.ndarray = np.ones((height, width))
    box = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    return mask, box


def _gen_diamond_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate diamond mask. See _gen_rect_mask()."""
    del ratio  # Unused
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (
        (Y + X >= mid)
        * (Y - X >= -mid)
        * (Y + X <= size + mid)
        * (Y - X <= mid)
    )
    return mask, [(0, mid), (mid, 0), (size - 1, mid), (mid, size - 1)]


def _gen_circle_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate circle mask. See _gen_rect_mask()."""
    del ratio  # Unused
    Y, X = np.ogrid[:size, :size]
    center = round(size / 2)  # center is also radius
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask = dist_from_center <= center
    return mask, [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)]


def _gen_triangle_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate triangle mask. See _gen_rect_mask()."""
    height = round(ratio * size)
    mid = round(size / 2)
    Y, X = np.ogrid[:height, :size]
    mask = (Y / height + 2 * X / size >= 1) * (Y / height - 2 * X / size >= -1)
    return mask, [(mid, 0), (size - 1, height - 1), (0, height - 1)]


def _gen_triangle_inverted_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate inverted triangle mask. See _gen_rect_mask()."""
    height = round(ratio * size)
    mid = round(size / 2)
    Y, X = np.ogrid[:height, :size]
    mask = (Y / height - 2 * X / size <= 0) * (Y / height + 2 * X / size <= 2)
    return mask, [(0, 0), (size - 1, 0), (mid, height - 1)]


def _gen_pentagon_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate pentagon mask. See _gen_rect_mask()."""
    del ratio  # Unused
    mid = round(size / 2)
    Y, X = np.ogrid[:size, :size]
    mask = (Y + X >= mid) * (Y - X >= -mid)
    return mask, [
        (0, mid),
        (size - 1, mid),
        (size - 1, size - 1),
        (0, size - 1),
    ]


def _gen_octagon_mask(
    size: int, ratio: float = 1.0
) -> tuple[np.ndarray, _KeyPoints]:
    """Generate octagon mask. See _gen_rect_mask()."""
    del ratio  # Unused
    edge = round((2 - np.sqrt(2)) / 2 * size)
    Y, X = np.ogrid[:size, :size]
    mask = (
        (Y + X >= edge)
        * (Y - X >= -(size - edge))
        * (Y + X <= 2 * size - edge)
        * (Y - X <= (size - edge))
    )
    return mask, [
        (edge, 0),
        (size - 1, edge),
        (size - edge, size - 1),
        (0, size - edge),
    ]


def gen_sign_mask(
    shape: str = "circle",
    hw_ratio: float = 1.0,
    obj_width_px: int = 64,
    use_box_mode: bool = False,
    pad_to_square: bool = True,
) -> tuple[torch.Tensor, _KeyPoints]:
    """Generate mask of object and source keypoints.

    The keypoints are a list of tuple (x, y) coordinates starting from the
    uppper left one and sorted clockwise. For example, keypoints of a
    rectangular mask are upper-left, upper-right, lower-right, and lower-left
    corners, respectively.

    Args:
        shape: Object shape defined based on classes in REAP.
        hw_ratio: Ratio of height over width of object.
        obj_width_px: Width of object in pixels.
        use_box_mode: If True, keypoints returned will be those of rectangular
            box around the object instead of the pre-defined keypoints.

    Returns:
        Binary mask and source keypoint for geometric transformation with
        respect to this mask.
    """
    shape_to_mask = {
        "circle": _gen_circle_mask,
        "triangle_inverted": _gen_triangle_inverted_mask,
        "triangle": _gen_triangle_mask,
        "rect": _gen_rect_mask,
        "diamond": _gen_diamond_mask,
        "pentagon": _gen_pentagon_mask,
        "octagon": _gen_octagon_mask,
        "square": _gen_rect_mask,
    }
    mask, box = shape_to_mask[shape](obj_width_px, ratio=hw_ratio)
    if use_box_mode:
        # Use mask from the correct shape but use keypoints of box
        _, box = _gen_rect_mask(obj_width_px, ratio=hw_ratio)

    mask = torch.from_numpy(mask)
    img_util.coerce_rank(mask, 4)
    pad_size = (obj_width_px, obj_width_px)
    mask, scales, padding = img_util.resize_and_pad(
        obj=mask,
        resize_size=pad_size,
        pad_size=pad_size if pad_to_square else None,
        is_binary=True,
        keep_aspect_ratio=True,
        return_params=True,
    )
    new_box = []
    for x, y in box:
        new_box.append((x * scales[1] + padding[0], y * scales[0] + padding[1]))
    return mask, new_box


def get_transform_fn(
    prob_geo: float | None = None,
    syn_rotate: float | None = None,
    syn_scale: float | None = None,
    syn_translate: float | None = None,
    syn_3d_dist: float | None = None,
    prob_colorjitter: float | None = None,
    syn_colorjitter: float | None = None,
    additive_uniform_noise: float | None = None,
    interp: str = "bilinear",
) -> tuple[TransformFn, TransformFn, TransformFn]:
    """Initialize geometric (for object and mask) and lighting transforms.

    When transforms are not applied, they are returned as identity function.

    Args:
        prob_geo: Probability of applying geometric transform.
        syn_rotate: Rotation degrees. Defaults to None (or 0 = no rotate).
        syn_scale: Scaling ratio. Defaults to None (or 1 = no scale).
        syn_translate: Translation distance.  Defaults to None.
        syn_3d_dist: 3D distortion. If syn_3d_dist is set to any non-None
            value, 3D or perspective transform will be used instead of
            affine transform. Defaults to None.
        prob_colorjitter: Probability of applying lighting transform.
        syn_colorjitter: Colorjitter intensity. Defaults to None (no color
            jitter or no lighting transform).
        interp: Interpolation mode. Defaults to "bilinear".

    Returns:
        Tuple of three transform functions: (i) geometric for object, (ii)
        geometric for mask, and (iii) lighting for object.
    """
    # Geometric transform
    geo_transform: TransformFn = nn.Identity()
    mask_transform: TransformFn = nn.Identity()

    if prob_geo is not None and prob_geo > 0:
        if syn_3d_dist is not None and syn_3d_dist > 0:
            transform_params = {
                "p": prob_geo,
                "distortion_scale": syn_3d_dist,
            }
            transform_fn = K.RandomPerspective
        else:
            transform_params = {
                "p": prob_geo,
                "degrees": syn_rotate,
                "translate": (syn_translate, syn_translate),
                "scale": None
                if syn_scale is None
                else (1 / syn_scale, syn_scale),
            }
            transform_fn = K.RandomAffine

        geo_transform = transform_fn(resample=interp, **transform_params)
        mask_transform = transform_fn(
            resample=kornia.constants.Resample.NEAREST, **transform_params
        )

    # Lighting transform (color jitter)
    light_transform: TransformFn = nn.Identity()
    if (
        prob_colorjitter is not None
        and prob_colorjitter > 0
        and syn_colorjitter is not None
        and syn_colorjitter > 0
    ):
        # Hue can't be change much; Otherwise, the color becomes wrong
        light_transform: TransformFn = K.ColorJiggle(
            brightness=syn_colorjitter,
            contrast=syn_colorjitter,
            saturation=syn_colorjitter,
            hue=0.05,
            p=prob_colorjitter,
        )
    if additive_uniform_noise is not None and additive_uniform_noise > 0:
        light_transform = nn.Sequential(
            light_transform, RandomNoise(max_range=additive_uniform_noise)
        )

    return (
        geo_transform,
        mask_transform,
        light_transform,
    )
