"""Optimizer utils."""

from __future__ import annotations

import logging
from typing import Any

import torch
from detectron2.config import instantiate
from detectron2.solver import LRMultiplier
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler
from detectron2.solver import build_optimizer as build_d2_optimizer
from torch import nn

logger = logging.getLogger(__name__)


def build_optimizer(cfg, model):
    """Build optimizer."""
    if cfg.MODEL.META_ARCHITECTURE == "YOLOF":
        logger.info("Using YOLOF optimizer.")
        return _build_yolof_optimizer(cfg, model)
    if "detrex" in cfg.MODEL.META_ARCHITECTURE:
        logger.info("Using detrex optimizer.")
        return _build_detrex_optimizer(cfg, model)
    return build_d2_optimizer(cfg, model)


def build_lr_scheduler(cfg, optimizer):
    """Build lr scheduler."""
    if "detrex" in cfg.MODEL.META_ARCHITECTURE:
        logger.info("Using detrex lr scheduler.")
        scheduler = _build_detrex_lr_scheduler(cfg, optimizer)
        # detrex scheduler has to be wrapped in LRMultiplier to be compatible
        # with DetectionCheckpointer (looking for state.dict()).
        scheduler = LRMultiplier(
            optimizer,
            scheduler,
            cfg.SOLVER.MAX_ITER,
            last_iter=-1,
        )
        return scheduler
    return build_d2_lr_scheduler(cfg, optimizer)


def _build_detrex_lr_scheduler(cfg, optimizer):
    _ = optimizer  # Unused
    return instantiate(cfg.lr_multiplier)


def _build_detrex_optimizer(cfg, model):
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    return optim


def _build_yolof_optimizer(cfg, model):
    norm_module_types = (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)

    params: list[dict[str, Any]] = []
    memo: set[torch.nn.parameter.Parameter] = set()
    for name, module in model.named_modules():
        for _, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in name:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            params += [
                {"params": [value], "lr": lr, "weight_decay": weight_decay}
            ]

    optimizer = torch.optim.SGD(
        params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
    )
    return optimizer
