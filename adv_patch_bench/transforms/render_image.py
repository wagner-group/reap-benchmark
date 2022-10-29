"""Image wrapper to handle object rendering."""

import copy
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

import adv_patch_bench.utils.image as img_util
import kornia.augmentation as K
import pandas as pd
import torchvision
from adv_patch_bench.transforms import render_object, util
from adv_patch_bench.utils.types import (
    SizePx,
    ImageTensor,
    Target,
    TransformFn,
    ImageTensorDet,
)


def _resize_pad_keep_ratio(image, img_size, interp):
    raise NotImplementedError()
    h, w = image.shape[-2:]
    # FIXME: This code does not work correctly because gt is not
    # adjusted in the same way (still requires padding).
    # Resize and pad perturbed_image to self.img_size preseving
    # aspect ratio. This also handles images whose width is the
    # shorter side in the varying input size case.
    if w < img_size[1]:
        # If real width is smaller than desired one, then height
        # must be longer than width so we scale down by height ratio
        scale = img_size[0] / h
    else:
        scale = 1
    resized_size = (int(h * scale), int(w * scale))
    image = img_util.resize_and_pad(
        image,
        pad_size=img_size,
        resize_size=resized_size,
        interp=interp,
    )
    h, w = img_size
    h_pad = h - resized_size[0]  # TODO: divided by 2?
    w_pad = w - resized_size[1]
    return image, (h_pad, w_pad)


class RenderImage:
    """Image wrapper for rendering adversarial patch and synthetic objects."""

    def __init__(
        self,
        dataset: str,
        sample: Dict[str, Any],  # TODO: YOLO?
        img_df: pd.DataFrame,
        img_size: Optional[SizePx] = None,
        img_mode: str = "BGR",
        interp: str = "bilinear",
        img_aug_prob_geo: Optional[float] = None,
        device: Any = "cuda",
        is_detectron: bool = True,
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
        self.filename: str = img_df["filename"]
        self._dataset: str = dataset
        self._img_df: pd.DataFrame = img_df
        self._interp: str = interp
        self._device: Any = device
        self._is_detectron: bool = is_detectron

        # Expect int image [0-255] and resize if size does not match img_size
        image: ImageTensor = sample["image"].float() / 255
        self.img_size: SizePx = (
            image.shape[-2:] if img_size is None else img_size
        )
        self.pad_size: SizePx = (0, 0)
        if self.img_size != image.shape[-2:]:
            image, pad_size = self._resize_image(image)
            self.pad_size = pad_size
        self.image: ImageTensor = image.to(device)

        # Copy metadata and use as target
        target = copy.deepcopy(sample)
        target.pop("image")
        self.target: Target = target

        if img_mode not in ("BGR", "RGB"):
            raise ValueError(
                f"Invalid img_mode {img_mode}! Must either be BGR or RGB."
            )
        self.img_mode: str = img_mode
        if img_mode == "BGR":
            self.image = self.image.flip(0)

        self.img_size_orig: SizePx = (
            sample["height"],
            sample["width"],
        )
        self.size_ratio: Tuple[float, float] = (
            self.img_size[0] / self.img_size_orig[0],
            self.img_size[1] / self.img_size_orig[1],
        )

        self.obj_tf_dict: Dict[int, render_object.RenderObject] = OrderedDict()

        # Init augmentation transform for image
        self._aug_geo_img: TransformFn = util._identity
        if img_aug_prob_geo is not None and img_aug_prob_geo > 0:
            self._aug_geo_img = K.RandomResizedCrop(
                self.img_size,
                scale=(0.8, 1),
                p=img_aug_prob_geo,
                resample=interp,
            )

    def _resize_image(self, image: ImageTensor) -> Tuple[ImageTensor, SizePx]:
        """Resize or pad image to self.img_size.

        Args:
            image: Image tensor to resize or pad.

        Returns:
            image: Resized or padded image.
            pad_size: Tuple of top and left padding.
        """
        h, w = image.shape[-2:]
        pad_size: SizePx
        if w != self.img_size[1]:
            raise ValueError(
                f"image of shape {image.shape} is not compatible with img_size "
                f"{self.img_size}!"
            )
        if h > self.img_size[0]:
            # If actual height is larger than desired height, just resize
            # image and avoid padding
            image = img_util.resize_and_pad(
                image, resize_size=self.img_size, interp=self._interp
            )
            pad_size = (0, 0)
        else:
            # Otherwise, pad height
            image, padding = img_util.resize_and_pad(
                image, pad_size=self.img_size, return_padding=True
            )
            pad_size = (padding[1], padding[0])
        assert image.shape[-2:] == self.img_size, (
            f"Image shape is {image.shape} but img_size is {self.img_size}. "
            "Image resize went wrong!"
        )
        return image, pad_size

    def create_object(
        self,
        obj_id: Optional[int],
        robj_fn: Callable[..., render_object.RenderObject],
        robj_kwargs: Dict[str, Any],
    ) -> None:
        """Create a new RenderObject and add to obj_tf_dict.

        RenderObject represents a digital object (e.g., adversarial patch,
        synthetic object). Once added to RenderImage, it will be rendered on
        image after apply_objects is called.

        Args:
            obj_tf: Given RenderObject to be added to this RenderImage. It must
                exist in as one of the object ids in img_df.

        Raises:
            ValueError: obj_id already exists in obj_tf_dict.
        """
        if obj_id in self.obj_tf_dict:
            raise ValueError(f"obj_id {obj_id} already exists!")

        assign_obj_id: int
        obj_df: Optional[pd.DataFrame]
        if obj_id is None:
            # If obj_id not specified, use the largest id not yet taken
            if self.obj_tf_dict:
                assign_obj_id = max(self.obj_tf_dict.keys()) + 1
            else:
                assign_obj_id = 0
            obj_df = None
        else:
            if obj_id not in list(self._img_df["object_id"]):
                raise ValueError(f"Given obj_id {obj_id} is not in img_df!")
            assign_obj_id = obj_id
            obj_df = self._img_df[self._img_df["object_id"] == obj_id].squeeze()

        robj: render_object.RenderObject = robj_fn(
            dataset=self._dataset,
            obj_df=obj_df,
            filename=self.filename,
            img_size=self.img_size,
            img_size_orig=self.img_size_orig,
            img_hw_ratio=self.size_ratio,
            img_pad_size=self.pad_size,
            device=self._device,
            **robj_kwargs,
        )
        self.add_object(robj, assign_obj_id)

    def add_object(
        self,
        robj: render_object.RenderObject,
        obj_id: Optional[int] = None,
    ) -> None:
        """Add an existing RenderObject to obj_tf_dict.

        Args:
            obj_tf: Given RenderObject to be added to this RenderImage.

        Raises:
            ValueError: obj_tf is not an RenderObject instance.
        """
        if not isinstance(robj, render_object.RenderObject):
            raise ValueError(
                f"Given object ({robj}) is not an RenderObject instance!"
            )

        if obj_id in self.obj_tf_dict:
            raise ValueError(f"obj_id {obj_id} already exists!")

        if obj_id is None:
            obj_id = max(self.obj_tf_dict.keys()) + 1

        self.obj_tf_dict[obj_id] = robj

    def update_object(
        self,
        robj: render_object.RenderObject,
        obj_id: int,
    ) -> None:
        """Replace RenderObject in self.obj_tf_dict at obj_id with robj.

        Args:
            robj: RenderObject to replace an existing one.
            obj_id: Object id to replace.

        Raises:
            ValueError: obj_id does not exist in self.obj_tf_dict.
        """
        if obj_id not in self.obj_tf_dict:
            raise ValueError(
                f"obj_id {obj_id} does not exist! Use add_object() instead."
            )
        self.obj_tf_dict[obj_id] = robj

    def get_object(
        self,
        obj_id: Optional[int] = None,
    ) -> render_object.RenderObject:
        """Get a RenderObject with a given obj_id.

        Args:
            obj_id: Object ID of RenderObject to retrieve. If None, obj_tf_dict
            must have exactly one object, and that object will be returned.
            Defaults to None.

        Raises:
            ValueError: obj_tf_dict does not have exactly 1 object.
            ValueError: Given obj_id does not exist.

        Returns:
            RenderObject with given obj_id.
        """
        if obj_id is None:
            num_objs = len(self.obj_tf_dict)
            if num_objs != 1:
                raise ValueError(
                    f"There are 0 or more than 1 objects ({num_objs}) in "
                    "obj_tf_dict. obj_id must be specified!"
                )
            obj_id: int = list(self.obj_tf_dict.keys())[0]

        if obj_id not in self.obj_tf_dict:
            raise ValueError(f"obj_id {obj_id} does not exist!")
        return self.obj_tf_dict[obj_id]

    def apply_objects(self) -> Tuple[ImageTensor, Target]:
        """Apply all RenderObjects in obj_tf_list to image.

        Returns:
            Image with patches and other objects applied.
        """
        image: ImageTensor = self.image
        target: Target = self.target

        for obj_tf in self.obj_tf_dict.values():
            image, target = obj_tf.apply_object(image, target)

        # Apply augmentation on the entire image
        image = self._aug_geo_img(image)
        image = img_util.coerce_rank(image, 3)

        return image, target

    def post_process_image(
        self,
        image: Optional[ImageTensor] = None,
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
        if image is None:
            image = self.image
        if image.ndim != 3:
            raise ValueError(
                f"image must have rank 3, but its shape is {image.shape}!"
            )
        if image.max() > 1 or image.min() < 0:
            raise ValueError("Pixel values of image are not between 0 and 1!")

        if not self._is_detectron:
            return image

        if self.img_mode == "BGR":
            image = image.flip(0)
        image *= 255
        return image

    def save_image(
        self,
        save_dir: str,
        image: Optional[ImageTensor] = None,
        ext: str = "png",
    ) -> None:
        """Save image to save_dir.

        If image is None, use self.image instead, and file name is taken from
        self.filename followed by the specified extension.

        Args:
            save_dir: Directory to save image to.
            image: Image to save. If None, use self.image. Defaults to None.
            ext: Image extension. Defaults to "png".
        """
        if image is None:
            image = self.image
        torchvision.utils.save_image(
            self.image, os.path.join(save_dir, f"{self.filename}.{ext}")
        )
