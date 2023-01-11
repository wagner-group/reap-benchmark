"""RenderObject for synthetic objects."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from detectron2 import structures
from PIL import Image

import adv_patch_bench.transforms.util as tf_util
import adv_patch_bench.utils.image as img_util
from adv_patch_bench.transforms import render_object
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    BatchImageTensorRGBA,
    BatchMaskTensor,
    ImageTensor,
    MaskTensor,
    SizePx,
    Target,
    TransformFn,
    TransformParamFn,
)


class SynObject(render_object.RenderObject):
    """RenderObject for synthetic objects and adversarial patch."""

    def __init__(
        self,
        img_size: SizePx | None = None,
        syn_obj_path: str | None = None,
        syn_rotate: float | None = None,
        syn_scale: float | None = None,
        syn_translate: float | None = None,
        syn_3d_dist: float | None = None,
        syn_colorjitter: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize SynOject.

        Args:
            img_size: Desired image size in pixel.
            syn_obj_path: Path to load synthetic object.
            syn_rotate: Rotation degrees to use for simulating synthetic object
                rotation. Defaults to None.
            syn_scale: _description_. Defaults to None.
            syn_translate: _description_. Defaults to None.
            syn_3d_dist: _description_. Defaults to None.
            syn_colorjitter: _description_. Defaults to None.
        """
        super().__init__(dataset="synthetic", **kwargs)
        if img_size is None:
            raise ValueError("img_size must be specified for SynObject!")
        if syn_obj_path is None:
            raise ValueError("syn_obj_path must be specified for SynObject!")
        self._img_size: SizePx = img_size

        # Resize obj_mask to obj_size_px and pad to img_size
        resized_obj_mask: MaskTensor = img_util.resize_and_pad(
            self.obj_mask,
            pad_size=self._img_size,
            resize_size=self._obj_size_px,
            is_binary=True,
        )
        self._resized_obj_mask: BatchMaskTensor = img_util.coerce_rank(
            resized_obj_mask, 4
        )

        # Load synthetic object to apply and resize/pad in the same way as mask
        self._resized_syn_obj: BatchImageTensor = self._load_syn_obj(
            syn_obj_path
        )

        # Set up transforms for synthetic object
        transforms = tf_util.get_transform_fn(
            prob_geo=1.0,
            syn_rotate=syn_rotate,
            syn_scale=syn_scale,
            syn_translate=syn_translate,
            syn_3d_dist=syn_3d_dist,
            prob_colorjitter=1.0,
            syn_colorjitter=syn_colorjitter,
            interp=self._interp,
        )
        self._geo_transform: TransformParamFn = transforms[0]
        self._light_transform: TransformFn = transforms[2]

    def _resize_and_pad(self, obj: BatchImageTensor) -> BatchImageTensor:
        obj: BatchImageTensor = img_util.resize_and_pad(
            obj,
            pad_size=self._img_size,
            resize_size=self._obj_size_px,
            is_binary=False,
            interp=self._interp,
        )
        return obj

    def _load_syn_obj(self, syn_obj_path: str) -> ImageTensor:
        """Load synthetic object and its mask and resize appropriately.

        Args:
            syn_obj_path: Path to load synthetic object from in PNG format.

        Returns:
            Synthetic object tensor resized and padded to img_size.
        """
        # Load synthetic object in PNG to numpy and then torch
        obj_numpy: np.ndarray = (
            np.array(Image.open(syn_obj_path).convert("RGBA")) / 255
        )
        syn_obj: ImageTensor = (
            torch.from_numpy(obj_numpy[:, :, :-1])
            .float()
            .permute(2, 0, 1)
            .to(self._device)
        )
        syn_obj = img_util.coerce_rank(syn_obj, 4)

        # Verify aspect ratio of loaded syn_obj
        obj_hw_ratio: float = syn_obj.shape[-2] / syn_obj.shape[-1]
        if abs(self._hw_ratio - obj_hw_ratio) > 1e-3:
            raise ValueError(
                f"Aspect ratio of loaded object is {obj_hw_ratio:.4f}, but it "
                f"should be {obj_hw_ratio:.4f}!"
            )

        # Resize syn_obj to obj_size_px and pad to img_size
        resized_syn_obj: BatchImageTensor = self._resize_and_pad(syn_obj)

        # Verify that syn_obj and obj_mask have the same size after resized
        mask_size: SizePx = self._resized_obj_mask.shape[-2:]
        assert resized_syn_obj.shape[-2:] == mask_size, (
            "resized_syn_obj must have the same size as resized_obj_mask "
            f"({resized_syn_obj.shape[-2:]} vs {mask_size})!"
        )

        return resized_syn_obj

    @staticmethod
    def apply_objects(
        images: BatchImageTensor,
        targets: list[Target],
        adv_patch: BatchImageTensor,
        patch_mask: BatchMaskTensor,
        tf_params: dict[str, Any],
        suppress_aug: bool = False,
    ) -> tuple[ImageTensor, Target]:
        """Apply adversarial patch to image using REAP approach.

        Args:
            image: Image to apply patch to.
            target: Target labels (unmodified).

        Returns:
            final_img: Image with transformed patch applied.
            target: Target with synthetic object label added.
        """
        syn_obj: BatchImageTensor = tf_params["syn_obj"]
        obj_mask: BatchMaskTensor = tf_params["obj_mask"]
        obj_class: torch.Tensor = tf_params["obj_class"]
        light_transform = tf_params["light_transform"]
        geo_transform = tf_params["geo_transform"]
        resize_and_pad = tf_params["resize_and_pad"]

        if adv_patch is None or patch_mask is None:
            adv_obj: BatchImageTensor = syn_obj
        else:
            # Resize patch and its mask to full image size
            adv_patch = resize_and_pad(adv_patch)
            patch_mask = resize_and_pad(patch_mask)
            # Apply lighting transform
            if not suppress_aug:
                adv_patch = light_transform(adv_patch)
                if adv_patch.is_leaf:
                    adv_patch = adv_patch.clamp(0, 1)
                else:
                    adv_patch.clamp_(0, 1)
            adv_obj: BatchImageTensor = (
                patch_mask * adv_patch + (1 - patch_mask) * syn_obj
            )

        # TODO(feature): Add augmentation for the sign

        rgba_adv_obj: BatchImageTensorRGBA = torch.cat(
            [adv_obj, obj_mask], dim=1
        )

        # Apply geometric transform on syn obj together with patch
        if not suppress_aug:
            rgba_adv_obj = geo_transform(rgba_adv_obj)
        rgba_adv_obj.clamp_(0, 1)

        # Place transformed syn obj to image
        obj_mask = rgba_adv_obj[:, -1:, :, :]
        adv_obj = rgba_adv_obj[:, :-1, :, :]
        adv_img: BatchImageTensor = obj_mask * adv_obj + (1 - obj_mask) * images

        # Modify target to account for applied syn object
        targets: Target = _modify_syn_target(targets, obj_class, obj_mask)

        # DEBUG
        # import torchvision
        # torchvision.utils.save_image(adv_img, "temp.png")
        # import pdb
        # pdb.set_trace()

        return adv_img, targets

    def aggregate_params(self, params_dicts: list[dict[str, Any]]) -> None:
        """Append self transform params to params_dicts."""
        params = {
            "obj_mask": self._resized_obj_mask,
            "syn_obj": self._resized_syn_obj,
            "obj_class": torch.tensor(
                [self._obj_class],
                device=self._device,
                dtype=torch.long,
            ),
            "alpha": torch.ones(1, device=self._device),
            "beta": torch.zeros(1, device=self._device),
        }
        for name, value in params.items():
            if name in params_dicts:
                params_dicts[name].append(value)
            else:
                params_dicts[name] = [value]

        if "light_transform" not in params_dicts:
            params_dicts["light_transform"] = self._light_transform
        if "geo_transform" not in params_dicts:
            params_dicts["geo_transform"] = self._geo_transform
        if "resize_and_pad" not in params_dicts:
            params_dicts["resize_and_pad"] = self._resize_and_pad


def _modify_syn_target(
    targets: list[Target],
    obj_classes: list[int],
    final_obj_masks: BatchMaskTensor,
) -> list[Target]:
    new_targets = []
    for target, obj_class, obj_mask in zip(
        targets, obj_classes, final_obj_masks
    ):
        new_targets.append(_modify_syn_target_one(target, obj_class, obj_mask))
    return new_targets


def _modify_syn_target_one(
    target: Target, obj_class: int, final_obj_mask: MaskTensor
) -> Target:
    """Modify target to include the added synthetic object.

    Since we paste a new synthetic sign on image, we have to add in a new
    synthetic label/target to compute loss and metrics.

    Args:
        target: Original target labels.
        obj_class: Class of syn object to add.
        final_obj_mask: Object mask after transformed.

    Returns:
        Modified target labels.
    """
    # get top left and bottom right points
    bbox = img_util.mask_to_box(final_obj_mask > 0)
    bbox = [b.cpu().item() for b in bbox]

    # Copy target since we have to add a new object
    new_target = copy.deepcopy(target)

    # Create Boxes object in XYXY_ABS format
    y_min, x_min, h_obj, w_obj = bbox
    new_bbox = [x_min, y_min, x_min + w_obj, y_min + h_obj]
    tensor_bbox = torch.tensor([new_bbox])
    # Create new instance with only synthetic oject
    new_instances = structures.Instances(target["instances"].image_size)
    new_instances.gt_boxes = structures.Boxes(tensor_bbox)
    new_instances.gt_classes = torch.tensor([obj_class])

    # Assign new instance
    new_target["instances"] = new_instances

    # Also update annotations for visualization
    new_anno: dict[str, Any] = {
        "bbox": new_bbox,
        "category_id": obj_class,
        "bbox_mode": target["annotations"][0]["bbox_mode"],
    }
    new_target["annotations"] = [new_anno]

    return new_target
