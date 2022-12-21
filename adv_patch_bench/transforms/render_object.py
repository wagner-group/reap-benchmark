"""Base class that applies adversarial patch and other objects to images."""

from __future__ import annotations

from typing import Any

import numpy as np
from detectron2.data import MetadataCatalog

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms import util
from adv_patch_bench.utils.types import MaskTensor, SizePx
from hparams import INTERPS


class RenderObject:
    """Base class for rendering objects on an image.

    RenderObject is a base class that contains basic functionality and params
    for implementing different methods of rendering adversarial patch or
    synthetic objects onto an image.
    """

    def __init__(
        self,
        dataset: str = "reap",
        obj_class: int | None = None,
        # img_size: SizePx = (1536, 2048),
        obj_size_px: SizePx = SizePx((64, 64)),
        interp: str = "bilinear",
        device: Any = "cuda",
        use_box_mode: bool = False,
        **kwargs,
    ) -> None:
        """Base ObjectTF associated with one object in image.

        This object stores metadata of that object and applies given adversarial
        patch (or any other synthetic objects) to a given image.

        Args:
            dataset: Name of dataset being used.
            obj_df: Row of DataFrame that holds metadata of this object.
            filename: Image file name (required when obj_df is None).
            obj_class: Class of object (required when obj_df is None).
            img_size: Desired image size in pixel.
            img_size_orig: Original image size.
            img_hw_ratio: Ratios of new over old height and width.
            img_pad_size: Pad size (height and width) of image.
            obj_size_px: Object size in pixel.
            interp: Interpolation method. Must be among "nearest", "bilinear",
                and "bicubic".
            device: Device to place tensors on (e.g., patch, mask).
            patch_aug_params: Dict of augmentation params, used in addition to
                transforms for applying patch and object. Defaults to None.
            is_detectron: Whether Detectron2 model is used.

        Raises:
            ValueError: df_row does not have exactly 1 entry.
            ValueError: Given obj_class_name from df_row does not match any
                known label from given dataset.
        """
        _ = kwargs  # Unused

        # Check dataset
        metadata = MetadataCatalog.get(dataset)
        if metadata is None:
            raise ValueError(
                f"dataset {dataset} is unknown! New dataset must provide "
                "metadata in hparams.py."
            )
        self._metadata = metadata.get("obj_dim_dict")
        self._obj_class: int = obj_class

        # Check interp
        if interp not in INTERPS:
            raise ValueError(
                f"interp {interp} is unknown! Must be among {INTERPS}."
            )
        self._interp: str = interp
        self._device: Any = device

        self._obj_size_px: SizePx = obj_size_px
        # self.img_size: SizePx = img_size
        # self.img_size_orig: SizePx = img_size_orig
        # self.img_hw_ratio: Tuple[float, float] = img_hw_ratio
        # self.img_pad_size: SizePx = img_pad_size
        self._hw_ratio: tuple[float, float] = self._metadata["hw_ratio"][
            self._obj_class
        ]

        # Generate object mask and source points for geometric transforms
        mask_src = self._get_obj_mask(use_box_mode)
        self.obj_mask: MaskTensor = mask_src[0].to(device)
        self.src_points: np.ndarray = mask_src[1]

    @staticmethod
    def get_augmentation(patch_aug_params, interp):
        """Initialize augmentation for patch and object."""
        if patch_aug_params is None:
            patch_aug_params = {}
        transforms = util.get_transform_fn(
            prob_geo=patch_aug_params.get("aug_prob_geo"),
            syn_rotate=patch_aug_params.get("aug_rotate"),
            syn_scale=patch_aug_params.get("aug_scale"),
            syn_translate=patch_aug_params.get("aug_translate"),
            syn_3d_dist=None,
            prob_colorjitter=patch_aug_params.get("aug_prob_colorjitter"),
            syn_colorjitter=patch_aug_params.get("aug_colorjitter"),
            interp=interp,
        )
        return transforms

    def _get_obj_mask(
        self, use_box_mode: bool = False
    ) -> tuple[MaskTensor, np.ndarray]:
        """Generate binary object mask and corresponding source points.

        Returns:
            Object mask, source points for geometric transform.
        """
        shape: str = self._metadata["shape"][self._obj_class]
        obj_mask, src = util.gen_sign_mask(
            shape,
            self._hw_ratio,
            self._obj_size_px[1],
            use_box_mode=use_box_mode,
        )
        obj_mask = obj_mask.float()
        obj_mask = img_util.coerce_rank(obj_mask, 4)
        src = np.array(src, dtype=np.float32)
        return obj_mask, src

    # @abstractmethod
    # def apply_object(
    #     self,
    #     image: ImageTensor,
    #     target: Target,
    # ) -> Tuple[ImageTensor, Target]:
    #     """Abstract class. Should be implemeted to apply patch to image.

    #     Args:
    #         image: Image to apply patch (or any object) to.
    #         target: Target labels that may be modified if needed.

    #     Returns:
    #         image: Image with transformed patch applied.
    #         target: Modified target label.
    #     """
    #     return image, target
