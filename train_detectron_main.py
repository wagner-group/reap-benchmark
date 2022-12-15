"""Train script for Detectron2.

Code is adapted from train_net.py.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
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
from adv_patch_bench.utils.argparse import reap_args_parser, setup_detectron_cfg

logger = logging.getLogger("detectron2")


# Need cfg/config for launch. pylint: disable=redefined-outer-name
def _get_evaluator(cfg, dataset_name, output_folder=None):
    """Create evaluator."""
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, output_dir=output_folder)


# Need cfg/config for launch. pylint: disable=redefined-outer-name
def do_test(cfg, config, model):
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
def do_train(cfg, config, model):
    resume: bool = config["base"]["resume"]
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

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small
    # training loop
    # pylint: disable=missing-kwoa,too-many-function-args
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration %d", start_iter)
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

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
                do_test(cfg, config, model)
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
    cfg = setup_detectron_cfg(config)
    # Register data. This has to be called by every process.
    data_util.register_dataset(config["base"])

    model = build_model(cfg)
    logger.info("Model:\n%s", model)
    if config["base"]["eval_only"]:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=config["base"]["resume"]
        )
        return do_test(cfg, config, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, config, model)
    return do_test(cfg, config, model)


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