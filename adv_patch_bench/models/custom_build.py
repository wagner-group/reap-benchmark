import torch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

CUSTOM_META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
CUSTOM_META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """Custom build model built on detectron2's build_model.

    Build the whole model architecture, defined by
    ``cfg.MODEL.META_ARCHITECTURE``. Note that it does not load any weights
    from ``cfg``.
    """
    # Import our yolov7p to be registered in CUSTOM_META_ARCH_REGISTRY
    # pylint: disable=unused-import,import-outside-toplevel
    from adv_patch_bench.models import yolov6, yolov7, yolov7p
    _ = yolov6, yolov7, yolov7p  # Unused imports

    meta_arch = cfg.MODEL.META_ARCHITECTURE
    # Search our custom registry first
    if meta_arch in CUSTOM_META_ARCH_REGISTRY:
        model = CUSTOM_META_ARCH_REGISTRY.get(meta_arch)(cfg)
    else:
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
