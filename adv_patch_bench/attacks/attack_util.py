"""Simple utility functions for preparing attacks."""

from typing import Optional, Tuple

import pickle
import torch
import torchvision
from adv_patch_bench.utils.types import ImageTensor, MaskTensor, SizeMM, SizePx
from adv_patch_bench.attacks import patch_mask_util
from hparams import DEFAULT_PATH_DEBUG_PATCH


def prep_adv_patch(
    attack_type: str = "none",
    adv_patch_path: Optional[str] = None,
    patch_size_mm: Optional[Tuple[int, float, float]] = None,
    obj_size_px: Optional[SizePx] = None,
    obj_size_mm: Optional[SizeMM] = None,
) -> Tuple[Optional[ImageTensor], Optional[MaskTensor]]:
    """Load and prepare adversarial patch along with its mask.

    Args:
        attack_type (str, optional): Type of attack to run. Options are "none",
            "debug", "random", "load". Defaults to "none".
        adv_patch_path (Optional[str], optional): Path to pickle file containing
            adversarial patch and its mask. Defaults to None.
        patch_size_mm: Tuple (num_patch, height, width). Height and width are
            in millimeters, and num_patch must be int.
        obj_size_px: Size of object to place patch on in pixels.
        obj_size_mm: Size of object to place patch on in millimeters.

    Returns:
        Tuple of adversarial patch and patch mask.
    """
    if attack_type == "none":
        return None, None

    adv_patch: ImageTensor
    patch_mask: MaskTensor

    if attack_type == "load":
        if adv_patch_path is None:
            raise ValueError(
                'If attack_type is "load", adv_patch_path must be specified!'
            )
        adv_patch, patch_mask = pickle.load(open(adv_patch_path, "rb"))
        return adv_patch, patch_mask

    if patch_size_mm is None or obj_size_px is None or obj_size_mm is None:
        raise ValueError(
            "patch_size_mm, obj_size_px, obj_size_mm must be specified when "
            'attack_type is not "none" or "load".'
        )

    # Generate new patch mask from given sizes
    patch_mask = patch_mask_util.gen_patch_mask(
        patch_size_mm,
        obj_size_px,
        obj_size_mm,
    )

    if attack_type == "per-sign":
        return None, patch_mask

    if attack_type == "debug":
        # Load 'arrow on checkboard' patch if specified (for debug)
        debug_patch_path: str = DEFAULT_PATH_DEBUG_PATCH
        loaded_image: torch.Tensor = torchvision.io.read_image(debug_patch_path)
        adv_patch = loaded_image.float()[:3, :, :] / 255
    elif attack_type == "random":
        # Patch with uniformly random pixels between [0, 1]
        adv_patch = torch.rand((3,) + obj_size_px)

    return adv_patch, patch_mask
