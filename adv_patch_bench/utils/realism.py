from __future__ import annotations

import copy
import os

import numpy as np
import torch
from PIL import Image

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms import lighting_tf


def compute_relight_params(
    torch_image,
    sign_mask,
    relight_method: str,
    relight_params,
    obj_class,
    src,
    tgt,
) -> torch.Tensor:
    syn_obj_path = os.path.join(
        "attack_assets", "synthetic", f"{obj_class}.png"
    )
    obj_numpy = np.array(Image.open(syn_obj_path).convert("RGBA")) / 255
    syn_obj = torch.from_numpy(obj_numpy[:, :, :-1]).float().permute(2, 0, 1)
    syn_obj = img_util.coerce_rank(syn_obj, 4)
    obj_height, obj_width = syn_obj.shape[-2:]
    orig_height, orig_width = sign_mask.shape[-2:]
    relight_sign_mask = img_util.resize_and_pad(
        sign_mask,
        resize_size=(obj_height, obj_width),
        is_binary=True,
        keep_aspect_ratio=False,
    ).float()
    relight_params["obj_mask"] = relight_sign_mask
    src = copy.deepcopy(src)
    src[:, 0] *= obj_width / orig_width
    src[:, 1] *= obj_height / orig_height
    relight_params["src_points"] = src
    relight_params["tgt_points"] = tgt
    relight_params["transform_mode"] = "perspective"
    if "percentile" not in relight_method:
        relight_params["syn_obj"] = syn_obj

    # calculate relighting parameters
    coeffs = lighting_tf.compute_relight_params(
        torch_image,
        method=relight_method,
        **relight_params,
    )
    return coeffs, syn_obj
