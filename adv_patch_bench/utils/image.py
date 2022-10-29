"""Utility file for manipulating and preprocessing images."""

import json
import os
from os.path import join
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as T
from adv_patch_bench.utils.types import ImageTensorGeneric, SizePx

_PadSize = Tuple[int, int, int, int]


def coerce_rank(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape x *in-place* to ndim rank by adding/removing first singleton dim.

    Args:
        x: Tensor to reshape. Usually an image.
        ndim: Desired number of dimension/rank.

    Raises:
        ValueError: Desired rank/ndim cannot be achieved.

    Returns:
        torch.Tensor: Tensor x that is reshaped to desired rank or ndim.
    """
    if x is None:
        return x

    if x.ndim == ndim:
        return x

    ndim_diff = ndim - x.ndim
    if ndim_diff < 0:
        for _ in range(-ndim_diff):
            x.squeeze_(0)
        if x.ndim != ndim:
            raise ValueError("Can't coerce rank.")
        return x

    for _ in range(ndim_diff):
        x.unsqueeze_(0)
    if x.ndim != ndim:
        raise ValueError("Can't coerce rank.")
    return x


def mask_to_box(mask):
    """Get a binary mask and returns a bounding box: y0, x0, h, w."""
    mask = coerce_rank(mask, 2)
    if mask.sum() <= 0:
        raise ValueError("mask is all zeros!")
    y, x = torch.where(mask)
    y_min, x_min = y.min(), x.min()
    return y_min, x_min, y.max() - y_min, x.max() - x_min


def resize_and_pad(
    obj: ImageTensorGeneric,
    resize_size: Optional[SizePx] = None,
    pad_size: Optional[SizePx] = None,
    is_binary: bool = False,
    interp: str = "bilinear",
    return_padding: bool = False,
) -> Union[ImageTensorGeneric, Tuple[ImageTensorGeneric, _PadSize]]:
    """Resize obj to resize_size and then pad_size it to pad_size.

    Args:
        obj: Object or image tensor to resize and pad.
        resize_size: Size to resize obj to. Defaults to None (no resize).
        pad_size: Size to pad resized obj to. Defaults to None (no pad).
        is_binary: Whether to treat obj as binary values. If True, interp will
            be set to "nearest". Defaults to False.
        interp: Interpolation method. Defaults to "bilinear".
        return_padding: If True, return four padding sizes together with final
            resized/padded object. Defaults to False.

    Raises:
        NotImplementedError: Invalid interpolation mode.

    Returns:
        Resized and padded obj. If return_padding is True, additionally return
        padding size.
    """
    if resize_size is not None and resize_size != obj.shape[-2:]:
        if is_binary or interp == "nearest":
            interp = T.InterpolationMode.NEAREST
        elif interp == "bicubic":
            interp = T.InterpolationMode.BICUBIC
        elif interp == "bilinear":
            interp = T.InterpolationMode.BILINEAR
        else:
            raise NotImplementedError(f"Interp {interp} not supported!")
        obj = T.resize(obj, resize_size, interpolation=interp)
    else:
        resize_size = obj.shape[-2:]

    if pad_size is not None and resize_size != pad_size:
        # Compute pad size that centers obj
        top: int = (pad_size[0] - resize_size[0]) // 2
        left: int = (pad_size[1] - resize_size[1]) // 2
        padding = (
            max(0, left),  # left
            max(0, top),  # top
            max(0, pad_size[1] - resize_size[1] - left),  # right
            max(0, pad_size[0] - resize_size[0] - top),  # bottom
        )
        obj = T.pad(obj, padding)
    else:
        padding = (0, 0, 0, 0)

    if return_padding:
        return obj, padding
    return obj


# ================= Functions for extracting transformations ================ #
# TODO(enhancement): Improve documentation.


def load_annotation(label_path, image_key):
    with open(join(label_path, "{:s}.json".format(image_key)), "r") as fid:
        anno = json.load(fid)
    return anno


def get_image_files(path):
    image_keys = []
    for entry in os.scandir(path):
        if (
            entry.path.endswith(".jpg") or entry.path.endswith(".png")
        ) and entry.is_file():
            image_keys.append(entry.name)
    return image_keys


def pad_image(img, pad_size=0.1, pad_mode="constant", return_pad_size=False):
    height, width = img.shape[-2:]
    pad_size = (
        int(max(height, width) * pad_size)
        if isinstance(pad_size, float)
        else pad_size
    )

    if isinstance(img, np.ndarray):
        height, width = img.shape[0], img.shape[1]
        pad_size_tuple = ((pad_size, pad_size), (pad_size, pad_size)) + (
            (0, 0),
        ) * (img.ndim - 2)
        img_padded = np.pad(img, pad_size_tuple, mode=pad_mode)
    else:
        height, width = img.shape[img.ndim - 2], img.shape[img.ndim - 1]
        img_padded = T.pad(img, pad_size, padding_mode=pad_mode)

    if return_pad_size:
        return img_padded, pad_size
    return img_padded


def crop(img_padded, mask, pad, offset):
    """Crop a square bounding box of an object with a correcponding
    segmentation mask from an (padded) image.

    Args:
        img_padded (np.ndarray): Image of shape (height, width, channels)
        mask (np.ndarray): A boolean mask of shape (height, width)
        pad (float): Extra padding for the bounding box from each endpoint of
            the mask as a ratio of `max(height, width)` of the object
        offset (int): Offset in case the given image is already padded

    Returns:
        np.ndarray: Cropped image
    """
    coord = np.where(mask)
    ymin, ymax = coord[0].min(), coord[0].max()
    xmin, xmax = coord[1].min(), coord[1].max()
    # Make sure that bounding box is square
    width, height = xmax - xmin, ymax - ymin
    size = max(width, height)
    xpad, ypad = int((size - width) / 2), int((size - height) / 2)
    extra_obj_pad = int(pad * size)
    size += 2 * extra_obj_pad
    xmin += offset - xpad - extra_obj_pad
    ymin += offset - ypad - extra_obj_pad
    xmax, ymax = xmin + size, ymin + size
    return img_padded[ymin:ymax, xmin:xmax]


def img_numpy_to_torch(img):
    assert img.ndim == 3 and isinstance(img, np.ndarray)
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0


def get_box(mask, pad):
    coord = np.where(mask)
    ymin, ymax = coord[0].min(), coord[0].max()
    xmin, xmax = coord[1].min(), coord[1].max()
    # Make sure that bounding box is square
    width, height = xmax - xmin, ymax - ymin
    size = max(width, height)
    xpad, ypad = int((size - width) / 2), int((size - height) / 2)
    extra_obj_pad = int(pad * size)
    size += 2 * extra_obj_pad
    xmin -= xpad + extra_obj_pad
    ymin -= ypad + extra_obj_pad
    xmax, ymax = xmin + size, ymin + size
    return ymin, ymax, xmin, xmax


def draw_from_contours(img, contours, color=[0, 0, 255, 255]):
    if not isinstance(contours, list):
        contours = [contours]
    for contour in contours:
        if contour.ndim == 3:
            contour_coord = (contour[:, 0, 1], contour[:, 0, 0])
        else:
            contour_coord = (contour[:, 1], contour[:, 0])
        img[contour_coord] = color
    return img


def letterbox(im, new_shape=(640, 640), color=114, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[2:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = (
        new_shape[1] - new_unpad[1],
        new_shape[0] - new_unpad[0],
    )  # wh padding
    # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = T.resize(im, new_unpad)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = T.pad(im, [left, top, right, bottom], fill=color)
    return im, ratio, (dw, dh)
