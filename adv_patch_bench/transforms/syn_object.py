"""RenderObject for synthetic objects."""

import copy
from typing import Any, Dict, Optional, TypeVar, Tuple

import adv_patch_bench.transforms.util as util
import adv_patch_bench.utils.image as img_util
import numpy as np
import torch
from adv_patch_bench.transforms import render_object
from adv_patch_bench.utils.types import (
    ImageTensor,
    MaskTensor,
    SizePx,
    Target,
    TransformFn,
    TransformParamFn,
    BatchMaskTensor,
    BatchImageTensor,
)
from detectron2 import structures
from hparams import LABEL_LIST
from PIL import Image

_ImageOrMask = TypeVar("_ImageOrMask", ImageTensor, MaskTensor)


class SynObject(render_object.RenderObject):
    """RenderObject for synthetic objects and adversarial patch."""

    def __init__(
        self,
        syn_obj_path: str = "./circle.png",
        syn_rotate: Optional[float] = None,
        syn_scale: Optional[float] = None,
        syn_translate: Optional[float] = None,
        syn_3d_dist: Optional[float] = None,
        syn_colorjitter: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize SynOject.

        Args:
            syn_obj_path: Path to load synthetic object.
            syn_rotate: Rotation degrees to use for simulating synthetic object
                rotation. Defaults to None.
            syn_scale: _description_. Defaults to None.
            syn_translate: _description_. Defaults to None.
            syn_3d_dist: _description_. Defaults to None.
            syn_colorjitter: _description_. Defaults to None.
        """
        super().__init__(**kwargs)
        self._bg_class: int = len(LABEL_LIST[self._dataset]) - 1

        # Resize obj_mask to obj_size_px and pad to img_size
        resized_obj_mask: MaskTensor = img_util.resize_and_pad(
            self.obj_mask,
            pad_size=self.img_size,
            resize_size=self._obj_size_px,
            is_binary=True,
        )
        self.resized_obj_mask: BatchMaskTensor = img_util.coerce_rank(
            resized_obj_mask, 4
        )

        # Load synthetic object to apply and resize/pad in the same way as mask
        resized_syn_obj: ImageTensor = self._load_syn_obj(syn_obj_path)
        self.resized_syn_obj: BatchImageTensor = img_util.coerce_rank(
            resized_syn_obj, 4
        )

        # Set up transforms for synthetic object
        transforms = util.get_transform_fn(
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
        self._mask_transform: TransformFn = transforms[1]
        self._light_transform: TransformFn = transforms[2]

    def _resize_patch(
        self, patch_or_mask: _ImageOrMask, is_mask: bool
    ) -> _ImageOrMask:
        """Resize adversarial patch or mask.

        This function overrides _resize_patch() in RenderObject. For synthetic
        object, we want to also pad patch and mask to self.img_size.

        Args:
            patch_or_mask: Adversarial patch or mask to resize.
            is_mask: Whether patch_or_mask is mask.

        Returns:
            Resized patch_or_mask.
        """
        patch_or_mask: _ImageOrMask = img_util.coerce_rank(patch_or_mask, 3)
        # Resize to obj_size_px and pad to img_size
        patch_or_mask: _ImageOrMask = img_util.resize_and_pad(
            patch_or_mask,
            resize_size=self._obj_size_px,
            pad_size=self.img_size,
            is_binary=is_mask,
            interp=self._interp,
        )
        assert patch_or_mask.shape[-2:] == self.img_size, (
            f"Shapes of patch/mask ({patch_or_mask.shape}) and img_size "
            f"({self.img_size}) do not match! Something went wrong."
        )
        return patch_or_mask

    def _load_syn_obj(
        self,
        syn_obj_path: str,
    ) -> ImageTensor:
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

        # Verify aspect ratio of loaded syn_obj
        obj_hw_ratio: float = syn_obj.shape[1] / syn_obj.shape[2]
        if abs(self._hw_ratio - obj_hw_ratio) > 1e-3:
            raise ValueError(
                f"Aspect ratio of loaded object is {obj_hw_ratio:.4f}, but it "
                f"should be {obj_hw_ratio:.4f}!"
            )

        # Resize syn_obj to obj_size_px and pad to img_size
        resized_syn_obj: ImageTensor = img_util.resize_and_pad(
            syn_obj,
            pad_size=self.img_size,
            resize_size=self._obj_size_px,
            is_binary=False,
            interp=self._interp,
        )

        # Verify that syn_obj and obj_mask have the same size after resized
        mask_size: SizePx = self.resized_obj_mask.shape[-2:]
        assert resized_syn_obj.shape[-2:] == mask_size, (
            "resized_syn_obj must have the same size as resized_obj_mask "
            f"({resized_syn_obj.shape[-2:]} vs {mask_size})!"
        )

        return resized_syn_obj

    def _modify_detectron_target(
        self, target: Target, bbox: Tuple[float, float, float, float]
    ) -> Target:
        # Copy target since we have to add a new object
        new_target = copy.deepcopy(target)
        instances: structures.Instances = target["instances"]

        # Create Boxes object in XYXY_ABS format
        y_min, x_min, h_obj, w_obj = bbox
        new_bbox = [x_min, y_min, x_min + w_obj, y_min + h_obj]
        tensor_bbox = torch.tensor([new_bbox])
        # Create new instance with only synthetic oject
        new_instances = structures.Instances(instances.image_size)
        new_instances.gt_boxes = structures.Boxes(tensor_bbox)
        new_instances.gt_classes = torch.tensor([self._obj_class])

        # Concatenate new instance to existing one
        new_target["instances"] = structures.Instances.cat(
            [instances, new_instances]
        )

        # Also update annotations for visualization
        new_anno: Dict[str, Any] = {
            "bbox": new_bbox,
            "category_id": self._obj_class,
            "bbox_mode": target["annotations"][0]["bbox_mode"],
        }
        new_target["annotations"].append(new_anno)

        return new_target

    def _modify_yolo_target(
        self, target: Target, bbox: Tuple[float, float, float, float]
    ) -> Target:
        # TODO: Add new target for YOLO
        # [image_id, class, x1, y1, label_width, label_height, obj_id]
        assert isinstance(target, torch.Tensor)
        y_min, x_min, h_obj, w_obj = bbox
        h, w = self.img_size
        label = [
            target[0],  # Index of image in batch
            self._obj_class,
            (x_min + w_obj / 2) / w,  # relative center x
            (y_min + h_obj / 2) / h,  # relative center y
            w_obj / w,  # relative width
            h_obj / h,  # relative height
            -1,
        ]
        target = torch.cat((target, torch.tensor(label).unsqueeze(0)))
        return target

    def _modify_syn_target(
        self, target: Target, final_obj_mask: MaskTensor
    ) -> Target:
        """Modify target to include the added synthetic object.

        Since we paste a new synthetic sign on image, we have to add in a new
        synthetic label/target to compute loss and metrics.

        Args:
            target: Original target labels.
            final_obj_mask: Object mask after transformed.

        Returns:
            Modified target labels.
        """
        # get top left and bottom right points
        bbox = img_util.mask_to_box(final_obj_mask == 1)
        bbox = [b.cpu().item() for b in bbox]

        if self._is_detectron:
            return self._modify_detectron_target(target, bbox)
        return self._modify_yolo_target(target, bbox)

    def apply_object(
        self,
        image: ImageTensor,
        target: Target,
    ) -> Tuple[ImageTensor, Target]:
        """Apply synthetic object to image and modify target accordingly.

        Args:
            image: Image to apply synthetic object (and patch) to.
            target: Target labels to be modified.

        Returns:
            adv_img: Image with object applied.
            target: Target with synthetic object label added.
        """
        image: BatchImageTensor = img_util.coerce_rank(image.clone(), 4)

        adv_obj: BatchImageTensor
        if self.adv_patch is not None and self.patch_mask is not None:
            # Apply adv patch if it has been loaded
            adv_patch: BatchImageTensor = img_util.coerce_rank(
                self.adv_patch.clone(), 4
            )
            patch_mask: BatchImageTensor = img_util.coerce_rank(
                self.patch_mask.clone(), 4
            )
            adv_obj = (
                patch_mask * adv_patch + (1 - patch_mask) * self.resized_syn_obj
            )
        else:
            adv_obj = self.resized_syn_obj

        # Apply geometric transform on syn obj together with patch
        adv_obj, tf_params = self._geo_transform(adv_obj)
        adv_obj.clamp_(0, 1)
        # Apply lighting transform
        if self._light_transform is not None:
            adv_obj = self._light_transform(adv_obj)
            adv_obj.clamp_(0, 1)

        # Apply the same geometric transform to obj mask
        obj_mask: BatchMaskTensor = self._mask_transform.apply_transform(
            self.resized_obj_mask, None, transform=tf_params.to(image.device)
        )
        # Place transformed syn obj to image
        adv_bimg: BatchImageTensor = obj_mask * adv_obj + (1 - obj_mask) * image
        adv_img: ImageTensor = img_util.coerce_rank(adv_bimg, 3)

        # Modify target to account for applied syn object
        target: Target = self._modify_syn_target(target, obj_mask)

        return adv_img, target
