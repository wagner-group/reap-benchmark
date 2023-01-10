"""Image wrapper to handle object rendering."""

from __future__ import annotations

import logging
import os
from typing import Any

import kornia.augmentation as K
import torch
import torchvision

from adv_patch_bench.transforms import (
    mtsd_object,
    reap_object,
    syn_object,
    util,
)
from adv_patch_bench.transforms.render_object import RenderObject
from adv_patch_bench.utils.types import (
    BatchImageTensor,
    BatchMaskTensor,
    DetectronSample,
    ImageTensor,
    ImageTensorDet,
    Target,
    TransformFn,
)

logger = logging.getLogger(__name__)


class RenderImage:
    """Image wrapper for rendering adversarial patch and synthetic objects."""

    def __init__(
        self,
        dataset: str,
        samples: list[dict[str, Any]],
        mode: str = "reap",
        obj_class: int | None = None,
        img_mode: str = "BGR",
        interp: str = "bilinear",
        img_aug_prob_geo: float | None = None,
        device: Any = "cuda",
        bg_class: int | None = None,
        robj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RenderImage containing full image and various metadata.

        Args:
            dataset: Name of dataset being used.
            sample: Sample to wrap RenderImage around. Must be a dictionary
                containing image and target/metadata.
            img_size: Desired image size. If None, use size of the image given
                in sample. Defaults to None.
            img_mode: Order of the color channel. Should either be "RGB" or
                "BGR" (default).
            interp: Resample interpolation method. Defaults to "bilinear".
            img_aug_prob_geo: Probability of applying addition augmentation to
                image when rendering. Defaults to None.
            device: Default device to place objects, patches, and masks.
                Defaults to "cuda".
            is_detectron: Whether we are using Detectron2. Defaults to True.

        Raises:
            ValueError: Invalid img_mode.
        """
        self._interp: str = interp
        self._is_detectron: bool = "instances" in samples[0]

        if img_mode not in ("BGR", "RGB"):
            raise ValueError(
                f"Invalid img_mode {img_mode}! Must either be BGR or RGB."
            )
        self.img_mode: str = img_mode
        self._mode: str = mode
        self.obj_classes: list[int] = []
        self.obj_ids: list[str] = []

        if robj_kwargs is None:
            robj_kwargs = {}

        # Expect int image [0-255] and resize if size does not match img_size
        images: list[ImageTensor] = []
        self.tf_params: dict[str, torch.Tensor | Any] = {}
        obj_to_img = []
        self._robj_fn = {
            "reap": reap_object.ReapObject,
            "mtsd": mtsd_object.MtsdObject,
            "synthetic": syn_object.SynObject,
        }[mode]

        if mode not in ("reap", "mtsd", "synthetic"):
            raise NotImplementedError(f"{mode} mode is not implemented!")

        for i, sample in enumerate(samples):
            image: ImageTensor = sample["image"].float() / 255
            # image = self._resize_image(image)
            image = image.flip(0) if img_mode == "BGR" else image
            file_name = sample["file_name"].split("/")[-1]
            images.append(image.to(device))

            temp_num_objs = len(self.obj_classes)
            is_obj_kept = []
            for oid, obj in enumerate(sample["annotations"]):
                is_obj_kept.append(False)
                cat_id = obj["category_id"]
                wrong_class = cat_id == bg_class or (
                    cat_id != obj_class and obj_class >= 0
                )
                # Skip obj of wrong class or has no REAP annotation in REAP mode
                if (mode == "reap" and not obj["has_reap"]) or wrong_class:
                    continue
                if (mode != "synthetic") and any(
                    point[2] != 2 for point in obj["keypoints"]
                ):
                    continue
                self.obj_classes.append(cat_id)
                robj: RenderObject = self._robj_fn(
                    obj_dict=obj,
                    dataset=dataset,
                    obj_class=obj["category_id"],
                    device=device,
                    image=image,
                    img_size=image.shape[-2:],
                    **robj_kwargs,
                )
                robj.aggregate_params(self.tf_params)
                obj_to_img.append(i)
                self.obj_ids.append(f"{file_name}-{oid}")
                is_obj_kept[-1] = True

            # Filter objs in sample by oids
            sample["instances"] = sample["instances"][is_obj_kept].to(device)
            sample["annotations"] = [
                anno
                for anno, is_kept in zip(sample["annotations"], is_obj_kept)
                if is_kept
            ]

            if temp_num_objs == len(self.obj_classes):
                logger.warning(
                    "No valid object is found in image %d/%d. Consider "
                    "removing this image.",
                    i,
                    len(samples),
                )

        self.images = torch.stack(images, dim=0).to(device, non_blocking=True)
        self.samples = samples
        self.num_objs: int = len(self.obj_classes)
        self.file_names: list[str] = [
            s["file_name"].split("/")[-1] for s in samples
        ]

        for name, params in self.tf_params.items():
            if isinstance(params, list) and isinstance(params[0], torch.Tensor):
                self.tf_params[name] = torch.cat(params, dim=0)
        self.tf_params["obj_transforms"] = RenderObject.get_augmentation(
            robj_kwargs.get("patch_aug_params"), interp
        )
        self.tf_params["obj_to_img"] = torch.tensor(obj_to_img, device=device)
        self.tf_params["interp"] = interp

        # Init augmentation transform for image
        self._aug_geo_img: TransformFn = util.identity
        if img_aug_prob_geo is not None and img_aug_prob_geo > 0:
            self._aug_geo_img = K.RandomResizedCrop(
                self.images.shape[-2:],
                scale=(0.8, 1),
                p=img_aug_prob_geo,
                resample=interp,
            )

    def _slice_images_and_params(
        self, obj_indices: torch.Tensor
    ) -> tuple[BatchImageTensor, list[DetectronSample], dict[str, Any]]:
        if obj_indices is None:
            return self.images, self.samples, self.tf_params
        tf_params = {}
        for key, val in self.tf_params.items():
            if isinstance(val, torch.Tensor):
                tf_params[key] = val.index_select(0, obj_indices)
            else:
                tf_params[key] = val
        obj_to_img = tf_params["obj_to_img"]
        img_indices = torch.unique(obj_to_img)
        for i, img_idx in enumerate(img_indices):
            # This should be correct because torch.unique() sorts values
            obj_to_img[obj_to_img == img_idx] = i
        images = self.images.index_select(0, img_indices)
        samples = [self.samples[int(i)] for i in img_indices]
        return images, samples, tf_params

    def apply_objects(
        self,
        adv_patch: BatchImageTensor | None = None,
        patch_mask: BatchMaskTensor | None = None,
        obj_indices: list[int] | None = None,
        suppress_aug: bool = False,
    ) -> tuple[BatchImageTensor, list[Target]]:
        """Render adversarial patches (or objects) on image.

        This calls apply_object() on each of RenderObjects in obj_tf_dict.

        Returns:
            image: Image with patches (or synthetic objects) applied.
            target: Updated target labels to account for applied object if any.
        """
        images, samples, tf_params = self._slice_images_and_params(obj_indices)
        orig_images = images

        if adv_patch is None or patch_mask is None:
            return images, samples

        images, targets = self._robj_fn.apply_objects(
            images,
            samples,
            adv_patch,
            patch_mask,
            tf_params,
            suppress_aug=suppress_aug,
        )

        # Apply augmentation on the entire image
        if not suppress_aug:
            images = self._aug_geo_img(images)

        if images.isnan().any():
            logger.warning(
                "NaN value(s) found in rendered images! Returning originals..."
            )
            images = orig_images

        return images, targets

    def post_process_image(
        self,
        images: ImageTensor | None = None,
    ) -> ImageTensorDet:
        """Post-process image by puting it in original format and scale to 255.

        Args:
            image: Image to post-process. If None, use self.image instead.
                Defaults to None.

        Raises:
            ValueError: image does not have rank 3.
            ValueError: Pixel values are not betwee 0 and 1.

        Returns:
            Processed image.
        """
        if images is None:
            images = self.images
        if images.ndim != 4:
            raise ValueError(
                f"image must have rank 4, but its shape is {images.shape}!"
            )
        assert ((0 <= images) & (images <= 1)).all(), (
            "Pixel values of image are not between 0 and 1 (min: "
            f"{images.min().item():.2f}, max: {images.max().item():.2f})!"
        )

        if not self._is_detectron:
            return images

        if self.img_mode == "BGR":
            images = images.flip(1)
        images *= 255
        return images

    def save_images(
        self,
        save_dir: str,
        image: ImageTensor | None = None,
        name: str = "temp.png",
    ) -> None:
        """Save image to save_dir.

        If image is None, use self.image instead, and file name is taken from
        self.filename followed by the specified extension.

        Args:
            save_dir: Directory to save image to.
            image: Image to save. If None, use self.image. Defaults to None.
            name: Image name to save as. Defaults to "temp.png".
        """
        if image is None:
            image = self.images
        torchvision.utils.save_image(self.images, os.path.join(save_dir, name))
