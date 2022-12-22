"""Generate adversarial patch on Detectron2 model."""

from __future__ import annotations

import logging
import pathlib
import pickle
import random
from typing import Any

import detectron2
import numpy as np
import torch
import torchvision
import yaml
from detectron2.data import MetadataCatalog
from torch.backends import cudnn
from tqdm import tqdm

import adv_patch_bench.dataloaders.detectron.util as data_util
import adv_patch_bench.utils.argparse as args_util
from adv_patch_bench.attacks import attack_util, base_attack
from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import (
    DetectronSample,
    ImageTensor,
    MaskTensor,
    SizeMM,
    SizePx,
)
from hparams import MAPILLARY_IMG_COUNTS_DICT

logger = logging.getLogger(__name__)


def _collect_attack_rimgs(
    dataloader: Any,
    num_bg: int | float,
    obj_class: int | None = None,
    rimg_kwargs: dict[str, Any] | None = None,
    robj_kwargs: dict[str, Any] | None = None,
) -> render_image.RenderImage:
    """Collect background images to be used by the attack.

    Args:
        dataloader: Detectron data loader.
        num_bg: Num total background images to collect.
        obj_class: Desired obj class. If specified, only select images from
            class class_name. Defaults to None.
        filter_file_names: List of image file names to use as attack_bg_syn.
        rimg_kwargs: Keyword args for initializing RenderImage.
        robj_kwargs: Keyword args for initializing RenderObject.

    Returns:
        Background images in form of render_image.RenderImage.
    """
    if rimg_kwargs is None:
        raise ValueError("rimg_kwargs must not be specified!")
    if robj_kwargs is None:
        raise ValueError("robj_kwargs must not be specified!")

    if num_bg < 1:
        assert obj_class is not None
        print(f"num_bg is a fraction ({num_bg}).")
        # TODO(NewDataset): Made compatible with other datasets.
        num_bg = round(MAPILLARY_IMG_COUNTS_DICT[obj_class] * num_bg)
        print(f"For {obj_class}, this is {num_bg} images.")
    num_bg = int(num_bg)

    backgrounds: list[DetectronSample] = []
    print("=> Collecting background images...")
    for _, batch in enumerate(tqdm(dataloader)):
        backgrounds.extend(batch)
        if len(backgrounds) >= num_bg:
            break

    rimg: render_image.RenderImage = render_image.RenderImage(
        dataset="reap",
        samples=backgrounds,
        robj_kwargs=robj_kwargs,
        **rimg_kwargs,
    )

    logger.info(
        "=> %d backgrounds with %d objects collected.",
        len(rimg.images),
        rimg.num_objs,
    )
    assert len(rimg.images) == num_bg and rimg.num_objs >= num_bg
    return rimg


def _generate_adv_patch(
    model: torch.nn.Module,
    rimg: render_image.RenderImage,
    patch_size_mm: tuple[int, float, float] = (1, 200.0, 200.0),
    obj_size_mm: SizeMM = SizeMM((900.0, 900.0)),
    obj_size_px: SizePx = SizePx((64, 64)),
    save_images: bool = False,
    save_dir: pathlib.Path = pathlib.Path("./"),
    verbose: bool = False,
    config_attack: dict[str, Any] | None = None,
) -> tuple[ImageTensor, MaskTensor]:
    """Generate adversarial patch.

    Returns:
        Adversarial patch as torch.Tensor.
    """
    device = model.device
    _, patch_mask = attack_util.prep_adv_patch(
        attack_type="per-sign",
        patch_size_mm=patch_size_mm,
        obj_size_px=obj_size_px,
        obj_size_mm=obj_size_mm,
    )

    attack: base_attack.DetectorAttackModule = attack_util.setup_attack(
        config_attack=config_attack,
        is_detectron=True,
        model=model,
        verbose=verbose,
    )

    # Generate an adversarial patch
    adv_patch: ImageTensor = attack.run(
        rimg, patch_mask.to(device), batch_mode=False
    )
    adv_patch = adv_patch.detach().cpu().float()

    if save_images:
        torchvision.utils.save_image(
            patch_mask, str(save_dir / "patch_mask.png")
        )
        torchvision.utils.save_image(
            adv_patch, str(save_dir / "adversarial_patch.png")
        )

    return adv_patch, patch_mask


def main() -> None:
    """Main function for generating patch.

    Args:
        config: Config dict containing eval and attack config dicts.
    """
    config_attack: dict[str, dict[str, Any]] = config["attack"]
    config_atk_common: dict[str, Any] = config_attack["common"]
    # img_size: SizePx = config_base["img_size"]
    dataset: str = config_base["dataset"]
    split_file_path: str = config_base["split_file_path"]
    obj_class: int = config_base["obj_class"]
    synthetic: bool = config_base["synthetic"]
    save_dir: pathlib.Path = pathlib.Path(config_base["save_dir"])
    interp: str = config_base["interp"]
    num_bg: int | float = config_atk_common["num_bg"]
    class_name: str = MetadataCatalog.get(dataset).get("thing_classes")[
        obj_class
    ]

    # Set up model from config
    model = detectron2.engine.DefaultPredictor(cfg).model

    # Load data to use as background
    dataloader, _ = data_util.get_dataloader(config_base, sampler="shuffle")

    # Set up parameters for RenderImage and RenderObject
    rimg_kwargs: dict[str, Any] = {
        "img_mode": cfg.INPUT.FORMAT,
        "interp": interp,
        "img_aug_prob_geo": config_atk_common["img_aug_prob_geo"],
        "device": model.device,
        "obj_class": obj_class,
        "mode": "synthetic" if synthetic else "reap",
    }
    robj_kwargs: dict[str, Any] = {
        "obj_size_px": config_base["obj_size_px"],
        "interp": interp,
        "patch_aug_params": config_atk_common,
    }
    if synthetic:
        robj_kwargs = {
            **robj_kwargs,
            "syn_obj_path": config_base["syn_obj_path"],
            "syn_rotate": config_atk_common["syn_rotate"],
            "syn_scale": config_atk_common["syn_scale"],
            "syn_translate": config_atk_common["syn_translate"],
            "syn_3d_dist": config_atk_common["syn_3d_dist"],
            "syn_colorjitter": config_atk_common["syn_colorjitter"],
        }
    else:
        robj_kwargs = {
            **robj_kwargs,
            "reap_transform_mode": config_atk_common["reap_transform_mode"],
            "reap_use_relight": config_atk_common["reap_use_relight"],
        }

    # Collect background images for generating patch attack
    attack_rimg: render_image.RenderImage = _collect_attack_rimgs(
        dataloader,
        num_bg,
        obj_class=obj_class,
        rimg_kwargs=rimg_kwargs,
        robj_kwargs=robj_kwargs,
    )

    # Save background filenames in txt file if split_file_path was not given
    print("=> Saving names of images used to generate patch in txt file.")
    split_file_path = save_dir / f"{class_name}_attack_bg{num_bg}.txt"
    with split_file_path.open("w", encoding="utf-8") as file:
        for sample in attack_rimg.samples:
            file.write(f'{sample["file_name"].split("/")[-1]}\n')

    if config_base["debug"]:
        # Save all the background images
        rimg_save_dir = save_dir / "attack_bg_syn"
        rimg_save_dir.mkdir(exist_ok=True)
        attack_rimg.save_images(str(rimg_save_dir))

    # Generate mask and adversarial patch
    adv_patch, patch_mask = _generate_adv_patch(
        model=model,
        rimg=attack_rimg,
        patch_size_mm=config_base["patch_size_mm"],
        obj_size_mm=config_base["obj_size_mm"],
        obj_size_px=config_base["obj_size_px"],
        save_images=config_base["save_images"],
        save_dir=save_dir,
        verbose=config_base["verbose"],
        config_attack=config_attack,
    )

    # Save adv patch
    patch_path = save_dir / "adv_patch.pkl"
    print(f"Saving the generated adv patch to {patch_path}...")
    with patch_path.open("wb") as file:
        pickle.dump([adv_patch, patch_mask], file)

    # Save attack config
    patch_metadata_path = save_dir / "config.yaml"
    print(f"Saving the adv patch metadata to {patch_metadata_path}...")
    with patch_metadata_path.open("w", encoding="utf-8") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    config: dict[str, dict[str, Any]] = args_util.reap_args_parser(
        is_detectron=True, is_gen_patch=True, is_train=False
    )
    # Verify some args
    cfg = args_util.setup_detectron_cfg(config)
    config_base: dict[str, Any] = config["base"]
    seed: int = config_base["seed"]
    cudnn.benchmark = False
    cudnn.deterministic = True

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
