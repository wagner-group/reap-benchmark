"""Build custom model."""

import logging

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils import comm
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry
from torch.nn.parallel import DistributedDataParallel
from yolof.checkpoint import YOLOFCheckpointer

logger = logging.getLogger(__name__)

CUSTOM_META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
CUSTOM_META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def _build_detrex_model(cfg):
    model = instantiate(cfg.model)
    return model


def build_model(cfg):
    """Custom build model built on detectron2's build_model.

    Build the whole model architecture, defined by
    ``cfg.MODEL.META_ARCHITECTURE``. Note that it does not load any weights
    from ``cfg``.
    """
    # Import our custom models to be registered in CUSTOM_META_ARCH_REGISTRY
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if meta_arch == "YOLOF":
        # pylint: disable=unused-import,import-outside-toplevel
        from adv_patch_bench.models import yolof

        _ = yolof  # Unused imports
    elif meta_arch in ("YOLOV6", "YOLOV7", "YOLOV7P"):
        # pylint: disable=unused-import,import-outside-toplevel
        from adv_patch_bench.models import yolov6, yolov7, yolov7p

        _ = yolov6, yolov7, yolov7p  # Unused imports

    # Search our custom registry first
    if meta_arch in CUSTOM_META_ARCH_REGISTRY:
        model = CUSTOM_META_ARCH_REGISTRY.get(meta_arch)(cfg)
    elif "detrex" in meta_arch:
        model = _build_detrex_model(cfg)
    else:
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))

    if comm.get_world_size() > 1:
        if "detrex" in meta_arch:
            logger.info(
                "Using DDP for detrex with parameters: %s", str(cfg.train.ddp)
            )
            model = create_ddp_model(model, **cfg.train.ddp)
        else:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
            )

    _log_api_usage("modeling.meta_arch." + meta_arch)

    # Try to load checkpoint if exists (for evaluation only)
    if cfg.MODEL.META_ARCHITECTURE == "YOLOF":
        checkpointer_fn = YOLOFCheckpointer
    else:
        checkpointer_fn = DetectionCheckpointer
    checkpointer = checkpointer_fn(model, cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    return model
