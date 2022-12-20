"""Generate adversarial patch on Detectron2 model."""

from __future__ import annotations

import pathlib
import pickle
import random
from typing import Any, Callable

import detectron2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch.backends import cudnn
from tqdm import tqdm

import adv_patch_bench.dataloaders.detectron.util as data_util
import adv_patch_bench.utils.argparse as args_util
from adv_patch_bench.attacks import attack_util, base_attack, patch_mask_util
from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.dataloaders.detectron import (
    custom_build,
    custom_sampler,
    mapper,
)
from adv_patch_bench.transforms import reap_object, render_image, syn_object
from adv_patch_bench.utils.types import (
    DetectronSample,
    ImageTensor,
    MaskTensor,
    SizeMM,
    SizePx,
)
from hparams import LABEL_LIST, MAPILLARY_IMG_COUNTS_DICT


def collect_attack_rimgs(
    dataset: str,
    dataloader: Any,
    num_bg: int | float,
    robj_fn: Callable,
    anno_df: pd.DataFrame | None = None,
    class_name: str | None = None,
    filter_file_names: list[str] | None = None,
    rimg_kwargs: dict[str, Any] | None = None,
    robj_kwargs: dict[str, Any] | None = None,
) -> list[render_image.RenderImage]:
    """Collect background images to be used by the attack.

    Args:
        dataset: Name of dataset.
        dataloader: Detectron data loader.
        num_bg: Num total background images to collect.
        robj_fn: RenderObject constructor.
        anno_df: REAP annotation DataFrame. If specified, only select images
            belong to anno_df. Defaults to None.
        class_name: Desired class name. If specified, only select images from
            class class_name. Defaults to None.
        filter_file_names: List of image file names to use as attack_bg_syn.
        rimg_kwargs: Keyword args for initializing RenderImage.
        robj_kwargs: Keyword args for initializing RenderObject.

    Returns:
        attack_bgs: List of background images and their metadata, used by the
            attack.
        metadata: List of metadata of original background image, used as part of
            input to detectron model.
        backgrond: Numpy array of background images.
    """
    if rimg_kwargs is None:
        raise ValueError("rimg_kwargs must not be specified!")
    if robj_kwargs is None:
        raise ValueError("robj_kwargs must not be specified!")

    if num_bg < 1:
        assert class_name is not None
        print(f"num_bg is a fraction ({num_bg}).")
        # TODO(NewDataset): Made compatible with other datasets.
        num_bg = round(MAPILLARY_IMG_COUNTS_DICT[class_name] * num_bg)
        print(f"For {class_name}, this is {num_bg} images.")
    num_bg = int(num_bg)

    rimg_list: list[render_image.RenderImage] = []
    num_collected: int = 0
    print("=> Collecting background images...")

    for _, batch in enumerate(tqdm(dataloader)):
        file_name = batch[0]["file_name"]
        filename = file_name.split("/")[-1]

        # If split_file_path is specified, ignore other file names
        if filter_file_names is not None and filename not in filter_file_names:
            continue

        # If df is specified, ignore images that are not in df
        if anno_df is not None:
            img_df = anno_df[anno_df["filename"] == filename]
            if img_df.empty:
                continue
        else:
            img_df = None

        found: bool = False
        attack_obj_df: pd.DataFrame | None = None
        attack_obj_id: int | None = None
        if class_name is not None and img_df is not None:
            # If class_name is also specified, make sure that there is at least
            # one sign with label class_name in image. We use the first object
            # found.
            # TODO(AnnoObj): Implement get_object method when we create a new
            # annotation object so we avoid manually looping through DataFrame.
            for _, obj_df in img_df.iterrows():
                if obj_df["final_shape"] == class_name:
                    found = True
                    attack_obj_df = obj_df
                    break
            if attack_obj_df is None:
                continue
            attack_obj_id = attack_obj_df["object_id"]
        else:
            # No df provided or don't care about class
            found = True

        if found:
            # attack_obj_id is not used by synthetic attack
            rimg = render_image.RenderImage(
                dataset,
                batch[0],
                img_df,
                **rimg_kwargs,
            )
            rimg.create_object(attack_obj_id, robj_fn, robj_kwargs)
            rimg_list.append(rimg)
            num_collected += 1

        if num_collected >= num_bg:
            break

    print(f"=> {len(rimg_list)} backgrounds collected.")
    return rimg_list[:num_bg]


def _generate_adv_patch(
    model: torch.nn.Module,
    rimgs: list[render_image.RenderImage],
    patch_size_mm: tuple[int, float, float] = (1, 200.0, 200.0),
    obj_size_mm: SizeMM = SizeMM((900.0, 900.0)),
    obj_size_px: SizePx = SizePx((64, 64)),
    img_size: SizePx = SizePx((1536, 2048)),
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

    patch_mask: MaskTensor = patch_mask_util.gen_patch_mask(
        patch_size_mm,
        obj_size_px,
        obj_size_mm,
    )

    attack: base_attack.DetectorAttackModule = attack_util.setup_attack(
        config_attack=config_attack,
        is_detectron=True,
        model=model,
        input_size=img_size,
        verbose=verbose,
    )

    # Generate an adversarial patch
    adv_patch: ImageTensor = attack.run(rimgs, patch_mask.to(device))
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
    img_size: SizePx = config_eval["img_size"]
    dataset: str = config_eval["dataset"]
    split_file_path: str = config_eval["split_file_path"]
    obj_class: int = config_eval["obj_class"]
    synthetic: bool = config_eval["synthetic"]
    save_dir: pathlib.Path = pathlib.Path(config_eval["save_dir"])
    interp: str = config_eval["interp"]
    num_bg: int | float = config_atk_common["num_bg"]
    dataframe: pd.DataFrame = reap_util.load_annotation_df(
        config_eval["tgt_csv_filepath"]
    )
    class_name: str = LABEL_LIST[dataset][obj_class]

    # Set up model from config
    model = detectron2.engine.DefaultPredictor(cfg).model

    # Build dataloader
    data_dicts: list[DetectronSample] = detectron2.data.DatasetCatalog.get(
        config_eval["dataset"]
    )
    split_file_names: list[str] | None = None
    num_samples: int = len(data_dicts)
    if split_file_path is not None:
        print(f"Loading file names from {split_file_path}...")
        with open(split_file_path, "r", encoding="utf-8") as file:
            split_file_names = file.read().splitlines()
        # Update num samples
        num_samples = len(split_file_names)

    # pylint: disable=too-many-function-args
    dataloader = custom_build.build_detection_test_loader(
        data_dicts,
        mapper=mapper.BenignMapper(cfg, is_train=False),
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        # Use sampler for random sampling background images from new dataset,
        # but we recommend using a pre-defined file names in split_file_path.
        sampler=custom_sampler.ShuffleInferenceSampler(num_samples),
        pin_memory=True,
        split_file_names=split_file_names,
    )

    # Set up parameters for RenderImage and RenderObject
    rimg_kwargs: dict[str, Any] = {
        "img_size": img_size,
        "img_mode": "BGR",
        "interp": interp,
        "img_aug_prob_geo": config_atk_common["img_aug_prob_geo"],
        "is_detectron": True,
    }
    robj_kwargs: dict[str, Any] = {
        "obj_size_px": config_eval["obj_size_px"],
        "interp": interp,
        "patch_aug_params": config_atk_common,
    }
    if synthetic:
        robj_fn = syn_object.SynObject
        robj_kwargs = {
            **robj_kwargs,
            "syn_obj_path": config_eval["syn_obj_path"],
            "syn_rotate": config_atk_common["syn_rotate"],
            "syn_scale": config_atk_common["syn_scale"],
            "syn_translate": config_atk_common["syn_translate"],
            "syn_3d_dist": config_atk_common["syn_3d_dist"],
            "syn_colorjitter": config_atk_common["syn_colorjitter"],
            "is_detectron": True,
        }
    else:
        robj_fn = reap_object.ReapObject
        robj_kwargs = {
            **robj_kwargs,
            "reap_transform_mode": config_atk_common["reap_transform_mode"],
            "reap_use_relight": config_atk_common["reap_use_relight"],
        }

    # Collect background images for generating patch attack
    attack_rimgs: list[render_image.RenderImage] = collect_attack_rimgs(
        dataset,
        dataloader,
        num_bg,
        robj_fn,
        anno_df=dataframe,
        class_name=class_name,
        filter_file_names=split_file_names,
        rimg_kwargs=rimg_kwargs,
        robj_kwargs=robj_kwargs,
    )

    # Save background filenames in txt file if split_file_path was not given
    if split_file_names is None:
        print("=> Saving names of images used to generate patch in txt file.")
        split_file_path = save_dir / f"{class_name}_attack_bg{num_bg}.txt"
        with split_file_path.open("w", encoding="utf-8") as file:
            for rimg in attack_rimgs:
                file.write(f"{rimg.filename}\n")

    if config_eval["debug"]:
        # Save all the background images
        rimg_save_dir = save_dir / "attack_bg_syn"
        rimg_save_dir.mkdir(exist_ok=True)
        for rimg in attack_rimgs:
            rimg.save_images(str(rimg_save_dir))

    # Generate mask and adversarial patch
    adv_patch, patch_mask = _generate_adv_patch(
        model,
        rimgs=attack_rimgs,
        patch_size_mm=config_eval["patch_size_mm"],
        obj_size_mm=config_eval["obj_size_mm"],
        obj_size_px=config_eval["obj_size_px"],
        img_size=img_size,
        save_images=config_eval["save_images"],
        save_dir=save_dir,
        verbose=config_eval["verbose"],
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
        True, is_gen_patch=True
    )
    # Verify some args
    cfg = args_util.setup_detectron_cfg(config)
    config_eval: dict[str, Any] = config["eval"]
    seed: int = config_eval["seed"]
    cudnn.benchmark = True

    # Only allow reap or synthetic as dataset for generating patch
    if config_eval["dataset"] not in ("reap", "synthetic"):
        raise ValueError(
            "dataset must be either reap or synthetic, but it is "
            f"{config_eval['dataset']}!"
        )

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)

    # Register Detectron2 dataset
    data_util.register_dataset(config_eval)

    main()
