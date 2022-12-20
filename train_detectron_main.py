"""Train script for Detectron2.

Code is adapted from train_net.py.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from typing import Any

from packaging import version

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
import detectron2
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.build import (
    RepeatFactorTrainingSampler,
    get_detection_dataset_dicts,
)
from detectron2.engine import default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.events import EventStorage
from torch.nn.parallel import DistributedDataParallel

import adv_patch_bench.dataloaders.detectron.util as data_util
from adv_patch_bench.attacks import attack_util
from adv_patch_bench.dataloaders.detectron import mtsd_dataset_mapper
from adv_patch_bench.transforms.render_image import RenderImage
from adv_patch_bench.utils.argparse import reap_args_parser, setup_detectron_cfg

logger = logging.getLogger("detectron2")


def _get_sampler(cfg):
    """Define a custom process to get training sampler.

    This error is caused by torch.trunc raising a segfault (floating point
    exception) on pytorch docker image. Calling repeat_factors.long() before
    passing it to torch.trunc fixes this.
    """
    if cfg.DATALOADER.SAMPLER_TRAIN != "RepeatFactorTrainingSampler":
        return None
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=0,
        proposal_files=None,
    )
    repeat_factors = (
        RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    )
    # This line is the fix
    repeat_factors = repeat_factors.long().float()
    sampler = RepeatFactorTrainingSampler(repeat_factors)
    return sampler


# Need cfg/config for launch. pylint: disable=redefined-outer-name
def _get_evaluator(cfg, dataset_name, output_folder=None):
    """Create evaluator."""
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, output_dir=output_folder)


# Need cfg/config for launch. pylint: disable=redefined-outer-name
def evaluate(cfg, config, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        # pylint: disable=missing-kwoa,too-many-function-args
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = _get_evaluator(
            cfg,
            dataset_name,
            os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info(
                "Evaluation results for %s in csv format:", dataset_name
            )
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


# Need cfg/config for launch. pylint: disable=redefined-outer-name
def train(cfg, config, model, attack):
    config_base = config["base"]
    resume: bool = config_base["resume"]
    use_attack: bool = config_base["attack_type"] != "none"
    train_dataset = cfg.DATASETS.TRAIN[0]
    class_names = detectron2.data.MetadataCatalog.get(train_dataset).get(
        "thing_classes"
    )
    bg_class: int = len(class_names) - 1

    rimg_kwargs: dict[str, Any] = {
        "img_mode": cfg.INPUT.FORMAT,
        "interp": config_base["interp"],
        "img_aug_prob_geo": config_base["img_aug_prob_geo"],
        "device": model.device,
        "obj_class": config_base["obj_class"],
        "mode": "mtsd",
        "bg_class": bg_class,
    }
    robj_kwargs = {
        "obj_size_px": config_base["obj_size_px"],
        "interp": config_base["interp"],
        "reap_transform_mode": config_base["reap_transform_mode"],
        "reap_use_relight": config_base["reap_use_relight"],
    }

    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter)
        if comm.is_main_process()
        else []
    )

    # Create patch mask
    adv_patches, patch_masks = attack_util.prep_adv_patch_all_classes(
        dataset=train_dataset,
        attack_type=config_base["attack_type"],
        patch_size_mm=config_base["patch_size_mm"],
        obj_width_px=config_base["obj_size_px"][1],
    )
    for i, (adv_patch, patch_mask) in enumerate(zip(adv_patches, patch_masks)):
        if adv_patch is not None:
            adv_patches[i] = adv_patch.to(model.device)
        if patch_mask is not None:
            patch_masks[i] = patch_mask.to(model.device)

    sampler = _get_sampler(cfg)
    # pylint: disable=missing-kwoa,too-many-function-args
    data_loader = build_detection_train_loader(
        cfg,
        sampler=sampler,
        mapper=mtsd_dataset_mapper.MtsdDatasetMapper(cfg, is_train=True),
    )
    logger.info("Starting training from iteration %d", start_iter)
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            if use_attack:
                rimg: RenderImage = RenderImage(
                    dataset=config["base"]["dataset"],
                    samples=data,
                    robj_kwargs=robj_kwargs,
                    **rimg_kwargs,
                )
                if rimg.num_objs > 0:
                    cur_patch_mask = [patch_masks[i] for i in rimg.obj_classes]
                    cur_patch_mask = torch.cat(cur_patch_mask, dim=0)
                    assert len(cur_patch_mask) == rimg.num_objs
                    if config_base["attack_type"] == "per-sign":
                        cur_adv_patch = attack(
                            rimg, cur_patch_mask, batch_mode=True
                        )
                    else:
                        cur_adv_patch = [
                            adv_patches[i] for i in rimg.obj_classes
                        ]
                        cur_adv_patch = torch.cat(cur_adv_patch, dim=0)
                    img_render, data = rimg.apply_objects(
                        cur_adv_patch, cur_patch_mask
                    )
                    img_render = rimg.post_process_image(img_render)
                    for i, dataset_dict in enumerate(data):
                        dataset_dict["image"] = img_render[i]

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced
                )

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                evaluate(cfg, config, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


# Need cfg/config for launch. pylint: disable=redefined-outer-name
def main(config):
    """Main function."""
    cfg = setup_detectron_cfg(config, is_train=True)
    # Register data. This has to be called by every process.
    data_util.register_dataset(config["base"])

    model = build_model(cfg)
    logger.info("Model:\n%s", model)

    # TODO: no attack
    attack = attack_util.setup_attack(
        config_attack=config["attack"],
        is_detectron=True,
        model=model,
        input_size=config["base"]["img_size"],
        verbose=config["base"]["verbose"],
    )

    if config["base"]["eval_only"]:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=config["base"]["resume"]
        )
        return evaluate(cfg, config, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    train(cfg, config, model, attack)
    return evaluate(cfg, config, model)


if __name__ == "__main__":
    config: dict[str, dict[str, Any]] = reap_args_parser(
        True, is_gen_patch=False, is_train=True
    )
    launch(
        main,
        config["base"]["num_gpus"],
        num_machines=config["base"]["num_machines"],
        machine_rank=config["base"]["machine_rank"],
        dist_url=config["base"]["dist_url"],
        args=(config,),
    )
