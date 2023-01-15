"""Generate adversarial patch on Detectron2 model."""

from __future__ import annotations

import copy
import logging
import os
import random
import sys
import warnings
from typing import Any

import numpy as np
from packaging import version
from tqdm import tqdm

# Calling subprocess.check_output() with python version 3.8.10 or lower will
# raise NotADirectoryError. When torch calls this to call hipconfig, it does
# not catch this exception but only FileNotFoundError or PermissionError.
# This hack makes sure that correct exception is raised.
if version.parse(sys.version.split()[0]) <= version.parse("3.8.10"):
    import subprocess

    def _hacky_subprocess_fix(*args, **kwargs):
        raise FileNotFoundError(
            "Hacky exception. If this interferes with your workflow, consider "
            "using python >= 3.8.10 or simply try to comment this out."
        )

    subprocess.check_output = _hacky_subprocess_fix

# pylint: disable=wrong-import-position
import torch
import torchvision
from torch.backends import cudnn

import adv_patch_bench.dataloaders.detectron.util as data_util
import adv_patch_bench.utils.argparse as args_util
import adv_patch_bench.utils.image as img_util
from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.transforms import lighting_tf, util
from hparams import DATASET_METADATA

logger = logging.getLogger(__name__)
# This is to ignore a warning from detectron2/structures/keypoints.py:29
warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    """Main function for computing relighting params."""
    config_base["data_dir"] = os.path.join(
        config_base["data_dir"], "mapillary_vistas", "no_color"
    )
    data_dicts = data_util.get_dataset(config_base)
    hw_ratio_dict = DATASET_METADATA["reap"]["hw_ratio"]
    class_names_dict = DATASET_METADATA["reap"]["class_name"]
    obj_shape_dict = DATASET_METADATA["reap"]["shape"]
    relight_params = {"transform_mode": "perspective"}
    if RELIGHT_METHOD == "polynomial":
        relight_params["polynomial_degree"] = POLY_DEGREE
        relight_params["percentile"] = DROP_TOPK
        column_name = "poly_coeffs"
    elif RELIGHT_METHOD == "color_transfer":
        column_name = "ct_coeffs"

    anno_df = reap_util.load_annotation_df(config_base["tgt_csv_filepath"])
    anno_df = anno_df.assign(
        **{column_name: [None for _ in range(len(anno_df))]}
    )

    for data_dict in tqdm(data_dicts):
        file_name = data_dict["file_name"].split("/")[-1]
        # Load image and resize
        image = (
            torchvision.io.read_image(
                data_dict["file_name"], mode=torchvision.io.ImageReadMode.RGB
            )
            / 255
        )
        image = img_util.coerce_rank(image, 4)
        image = img_util.resize_and_pad(
            obj=image,
            resize_size=config_base["img_size"],
            pad_size=config_base["img_size"],
            keep_aspect_ratio=True,
        )

        # Loop through each object in image
        for anno in data_dict["annotations"]:
            obj_class = anno["category_id"]
            # Skip background (other) class
            if obj_class not in hw_ratio_dict:
                continue
            obj_id = anno["object_id"]
            obj_shape = obj_shape_dict[obj_class]
            # Generate sign mask and src points
            sign_mask, src = util.gen_sign_mask(
                obj_shape,
                hw_ratio=hw_ratio_dict[obj_class],
                obj_width_px=64,
                pad_to_square=False,
            )
            sign_mask = img_util.coerce_rank(sign_mask, 4)
            src = np.array(src).astype(np.float32)
            tgt = np.array(anno["keypoints"], dtype=np.float32)
            tgt = tgt.reshape(-1, 3)[:, :2]

            # Load synthetic object
            class_name = class_names_dict[obj_class]
            syn_obj_path = os.path.join(
                "attack_assets", "synthetic", f"{class_name}.png"
            )
            syn_obj = (
                torchvision.io.read_image(
                    syn_obj_path, mode=torchvision.io.ImageReadMode.RGB
                )
                / 255
            )
            syn_obj = img_util.coerce_rank(syn_obj, 4)
            obj_height, obj_width = syn_obj.shape[-2:]
            orig_height, orig_width = sign_mask.shape[-2:]

            # Resize sign/obj mask to match the size of syn object
            relight_sign_mask = img_util.resize_and_pad(
                sign_mask,
                resize_size=(obj_height, obj_width),
                is_binary=True,
                keep_aspect_ratio=False,
            ).float()
            relight_params["obj_mask"] = relight_sign_mask
            src = copy.deepcopy(src)
            # Resize src points to match the size of syn object
            src[:, 0] *= obj_width / orig_width
            src[:, 1] *= obj_height / orig_height
            relight_params["src_points"] = src
            relight_params["tgt_points"] = tgt
            relight_params["syn_obj"] = syn_obj

            # Calculate relighting parameters
            coeffs = lighting_tf.compute_relight_params(
                image,
                method=RELIGHT_METHOD,
                **relight_params,
            ).tolist()
            anno_df.loc[
                (anno_df["filename"] == file_name)
                & (anno_df["object_id"] == obj_id),
                column_name,
            ] = str(coeffs)

    anno_df.to_csv(config_base["tgt_csv_filepath"], index=False)


if __name__ == "__main__":
    RELIGHT_METHOD = "polynomial"  # "polynomial", "color_transfer"
    POLY_DEGREE = 1
    DROP_TOPK = 0.01

    config: dict[str, dict[str, Any]] = args_util.reap_args_parser(
        is_detectron=True, is_gen_patch=True, is_train=False
    )
    # Verify some args
    cfg = args_util.setup_detectron_cfg(config)
    config_base: dict[str, Any] = config["base"]
    seed: int = config_base["seed"]
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Set logging config
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(asctime)s - %(name)s - %(levelname)s]: %(message)s",
        level=logging.DEBUG
        if config["base"]["debug"] or config["base"]["verbose"]
        else logging.INFO,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Only allow reap or synthetic as dataset for generating patch
    if config_base["dataset"] not in ("reap", "synthetic"):
        raise ValueError(
            "dataset must be either reap or synthetic, but it is "
            f"{config_base['dataset']}!"
        )

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)

    # Register Detectron2 dataset
    data_util.register_dataset(config_base)

    main()
