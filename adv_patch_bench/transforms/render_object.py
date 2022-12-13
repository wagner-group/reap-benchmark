"""Base class that applies adversarial patch and other objects to images."""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import torch

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms import util
from adv_patch_bench.utils.types import (
    ImageTensor,
    MaskTensor,
    SizePx,
    Target,
    TransformFn,
    TransformParamFn,
)
from hparams import DATASETS, INTERPS, LABEL_LIST, OBJ_DIM_DICT

_ImageOrMask = TypeVar("_ImageOrMask", ImageTensor, MaskTensor)


class RenderObject:
    """Base class for rendering objects on an image.

    RenderObject is a base class that contains basic functionality and params
    for implementing different methods of rendering adversarial patch or
    synthetic objects onto an image.
    """

    def __init__(
        self,
        dataset: str = "reap",
        obj_df: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        obj_class: Optional[int] = None,
        img_size: SizePx = (1536, 2048),
        img_size_orig: SizePx = (1536, 2048),
        img_hw_ratio: Tuple[float, float] = (1.0, 1.0),
        img_pad_size: SizePx = (0, 0),
        obj_size_px: SizePx = (64, 64),
        interp: str = "bilinear",
        device: Union[torch.device, str] = "cuda",
        patch_aug_params: Optional[Dict[str, Any]] = None,
        is_detectron: bool = True,
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
        del kwargs  # Unused

        # TODO(feature): We should decouple model from transforms, but this
        # likely involves writing another wrapper for the models.
        self._is_detectron = is_detectron

        # Check dataset
        if dataset not in DATASETS:
            raise ValueError(
                f"dataset {dataset} is unknown! New dataset must provide "
                "metadata in hparams.py."
            )
        self._dataset: str = dataset

        # TODO(feature): Use a wrapped object instead of DataFrame
        self.obj_df: Optional[pd.DataFrame] = None
        self.filename: str
        self.obj_class_name: str
        self.obj_class: int
        if obj_df is None:
            if filename is None:
                raise ValueError(
                    "filename must be specified if obj_df is None!"
                )
            if obj_class is None:
                raise ValueError(
                    "obj_class must be specified if obj_df is None!"
                )
            self.filename = filename
            self.obj_class = obj_class
            self.obj_class_name = LABEL_LIST[dataset][obj_class]
        else:
            # Check that df_row has one entry
            if obj_df.ndim != 1:
                raise ValueError(
                    f"df_row must have exactly 1 entry (1 dim), but it has "
                    f"shape of {obj_df.shape}!"
                )
            self.obj_df = obj_df
            self.filename = obj_df["filename"]
            self.obj_class_name = obj_df["final_shape"]
            if self.obj_class_name not in LABEL_LIST[dataset]:
                raise ValueError(
                    f"Given obj_class_name ({self.obj_class_name}) does not "
                    "match any known label from given dataset!"
                )
            self.obj_class: int = LABEL_LIST[dataset].index(self.obj_class_name)

        # Check interp
        if interp not in INTERPS:
            raise ValueError(
                f"interp {interp} is unknown! Must be among {INTERPS}."
            )
        self._interp: str = interp
        self._device: Any = device

        self.obj_size_px: SizePx = obj_size_px
        self.img_size: SizePx = img_size
        self.img_size_orig: SizePx = img_size_orig
        self.img_hw_ratio: Tuple[float, float] = img_hw_ratio
        self.img_pad_size: SizePx = img_pad_size
        self.hw_ratio: Tuple[float, float] = OBJ_DIM_DICT[dataset]["hw_ratio"][
            self.obj_class
        ]

        # Generate object mask and source points for geometric transforms
        mask_src = self._get_obj_mask()
        self.obj_mask: MaskTensor = mask_src[0].to(device)
        self.src_points: np.ndarray = mask_src[1]

        # Default values of relighting transform
        self.alpha: torch.Tensor = img_util.coerce_rank(
            torch.tensor(1.0, device=device), 3
        )
        self.beta: torch.Tensor = img_util.coerce_rank(
            torch.tensor(0.0, device=device), 3
        )

        # Initialize augmentation for patch and object
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
            interp=self._interp,
        )
        self.aug_geo: TransformParamFn = transforms[0]
        self.aug_mask: TransformFn = transforms[1]
        self.aug_light: TransformFn = transforms[2]

        self.adv_patch: Optional[ImageTensor] = None
        self.patch_mask: Optional[MaskTensor] = None

    def load_adv_patch(
        self,
        adv_patch: Optional[ImageTensor] = None,
        patch_mask: Optional[MaskTensor] = None,
    ) -> None:
        """Load and prepare adversarial patch along with its mask.

        adv_patch and patch_mask do not need to be set at the same time, and
        either one can be updated later. Both patch and mask will be resized to
        obj_size_px.

        Args:
            adv_patch: Adversarial patch to apply. Must have 3 channels (RGB).
            patch_mask: Corresponding binary mask of adv_patch with respect to
                the object to apply to.
        """
        self._load_adv_patch_base(adv_patch=adv_patch, patch_mask=patch_mask)

    def _resize_patch(
        self, patch_or_mask: _ImageOrMask, is_mask: bool
    ) -> _ImageOrMask:
        """Resize adv patch or mask.

        Args:
            patch_or_mask: Adversarial patch or mask to resize.
            is_mask: Whether patch_or_mask is mask.

        Returns:
            Resized patch_or_mask.
        """
        patch_or_mask: _ImageOrMask = img_util.coerce_rank(patch_or_mask, 3)
        # Resize to obj_size_px
        patch_or_mask: _ImageOrMask = img_util.resize_and_pad(
            patch_or_mask,
            resize_size=self.obj_size_px,
            is_binary=is_mask,
            interp=self._interp,
        )
        assert patch_or_mask.shape[-2:] == self.obj_size_px, (
            f"Shapes of patch/mask ({patch_or_mask.shape}) and obj_size_px "
            f"({self.obj_size_px}) do not match! Something went wrong."
        )
        return patch_or_mask

    def _load_adv_patch_base(
        self,
        adv_patch: Optional[ImageTensor] = None,
        patch_mask: Optional[MaskTensor] = None,
    ) -> None:
        """Load and prepare adversarial patch along with its mask.

        Set attributes adv_patch and patch_mask to given tensors (if not None).
        This is a base private method that can be called by actual
        load_adv_patch() which may be modified by other RenderObject classes.

        Args:
            adv_patch: Adversarial patch to apply. Must have 3 channels (RGB).
            patch_mask: Corresponding binary mask of adv_patch with respect to
                the object to apply to.
        """
        if adv_patch is not None:
            self.adv_patch = self._resize_patch(adv_patch, False).to(
                self._device
            )
        if patch_mask is not None:
            self.patch_mask = self._resize_patch(patch_mask, True).to(
                self._device
            )

    def _get_obj_mask(self) -> Tuple[MaskTensor, np.ndarray]:
        """Generate binary object mask and corresponding source points.

        Returns:
            Object mask, source points for geometric transform.
        """
        shape: str = OBJ_DIM_DICT[self._dataset]["shape"][self.obj_class]
        obj_mask, src = util.gen_sign_mask(
            shape, self.hw_ratio, self.obj_size_px[1]
        )
        obj_mask = torch.from_numpy(obj_mask).float()
        obj_mask = img_util.coerce_rank(obj_mask, 3)
        return obj_mask, src

    @abstractmethod
    def apply_object(
        self,
        image: ImageTensor,
        target: Target,
    ) -> Tuple[ImageTensor, Target]:
        """Abstract class. Should be implemeted to apply patch to image.

        Args:
            image: Image to apply patch (or any object) to.
            target: Target labels that may be modified if needed.

        Returns:
            image: Image with transformed patch applied.
            target: Modified target label.
        """
        return image, target
