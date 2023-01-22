"""Registers datasets, and defines other dataloading utilities."""

from __future__ import annotations

import os
from typing import Any

import detectron2
import numpy as np
import torch
import torchvision
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from yolof.data.dataset_mapper import YOLOFDtasetMapper
from yolof.data.detection_utils import transform_instance_annotations

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.dataloaders.detectron.mtsd_dataset_mapper import (
    get_mtsd_transforms,
)
from adv_patch_bench.transforms import util
from adv_patch_bench.utils.types import SizePx
from hparams import DATASET_METADATA, DEFAULT_SYN_OBJ_DIR


class MtsdYoloDatasetMapper(YOLOFDtasetMapper):
    """A callable which takes a dataset dict in Detectron2 Dataset format.

    This is the default callable to be used to map your dataset dict into
    training data.
    """

    def __init__(
        self,
        cfg: detectron2.config.CfgNode,
        config_base: dict[str, Any],
        is_train: bool = True,
        img_size: SizePx | None = None,
    ) -> None:
        """Initialize MTSD data mapper for YOLOF.

        Args:
            cfg: Detectron2 config.
            config_base: Base config.
            is_train: Whether we are training. Defaults to True.
            img_size: Resize all images to this fixed size instead of just
                loading the original size. Defaults to None.
        """
        # Init from config. pylint: disable=missing-kwoa,too-many-function-args
        super().__init__(cfg, is_train)
        self.use_keypoint = True

        # MTSD specific
        metadata = DATASET_METADATA["mtsd_no_color"]
        class_names = metadata["class_name"]
        hw_ratio_dict = metadata["hw_ratio"]
        shape_dict = metadata["shape"]

        self._img_size: SizePx = img_size
        self._syn_objs = {}
        self._syn_obj_masks = {}
        self._relight_params = {
            "method": config_base["reap_relight_method"],
            "percentile": config_base["reap_relight_percentile"],
            "transform_mat": torch.eye(3, dtype=torch.float32).view(1, 3, 3),
        }
        if "polynomial" in config_base["reap_relight_method"]:
            self._relight_params["polynomial_degree"] = config_base[
                "reap_relight_polynomial_degree"
            ]
        if "percentile" not in config_base["reap_relight_method"]:
            self._relight_params["interp"] = config_base["interp"]

        # Load dict of syn objs and masks
        for obj_class, class_name in class_names.items():
            if class_name == "other":
                continue
            obj_mask, _ = util.gen_sign_mask(
                shape_dict[obj_class],
                hw_ratio=hw_ratio_dict[obj_class],
                obj_width_px=config_base["obj_size_px"][1],
                pad_to_square=False,
            )
            syn_obj_path = os.path.join(
                DEFAULT_SYN_OBJ_DIR, "synthetic", f"{class_name}.png"
            )
            syn_obj = torchvision.io.read_image(
                syn_obj_path, mode=torchvision.io.ImageReadMode.RGB
            )
            syn_obj = syn_obj.float() / 255
            syn_obj = img_util.coerce_rank(syn_obj, 4)
            obj_mask = img_util.coerce_rank(obj_mask, 4)
            self._syn_objs[obj_class] = syn_obj
            self._syn_obj_masks[obj_class] = obj_mask.float()
        self._column_name = f'{self._relight_params["method"]}_coeffs'

    def _load_image_with_annos(self, dataset_dict):
        """Load the image and annotations given a dataset_dict."""
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )
        # EDIT: MTSD resize ================================================= #
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        scales, padding = [1, 1], [0] * 4
        if self._img_size is not None:
            image, scales, padding = img_util.resize_and_pad(
                obj=image,
                resize_size=self._img_size,
                pad_size=self._img_size,
                keep_aspect_ratio=True,
                return_params=True,
            )
            image = image.permute(1, 2, 0).numpy()
            dataset_dict["width"] = self._img_size[1]
            dataset_dict["height"] = self._img_size[0]
        # =================================================================== #
        utils.check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                # EDIT: Scale and shift bbox according to new image size ==== #
                xmin, ymin, xmax, ymax = anno["bbox"]
                ymin = ymin * scales[0] + padding[1]
                ymax = ymax * scales[0] + padding[1]
                xmin = xmin * scales[1] + padding[0]
                xmax = xmax * scales[1] + padding[0]
                anno["bbox"] = [xmin, ymin, xmax, ymax]
                # EDIT: Add transform info to anno ========================== #
                get_mtsd_transforms(
                    anno,
                    image,
                    self._column_name,
                    self._syn_objs,
                    self._syn_obj_masks,
                    self._relight_params,
                )
                # =========================================================== #

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return image, None

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other
            # types of data
            # apply meta_infos for mosaic transformation
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    add_meta_infos=self.add_meta_infos,
                )
                for obj in dataset_dict[
                    "annotations"
                ]  # Don't pop annotations yet; need it later.
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = None
        return image, annos

    def __call__(self, dataset_dict):
        """Map dataset_dict."""
        dataset_dict = super().__call__(dataset_dict)
        # if self.is_train:
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict
        instances = dataset_dict["instances"]
        new_annos = []
        num_instances = len(instances)
        for i in range(num_instances):
            obj = {
                "bbox": instances[i].gt_boxes.tensor[0].tolist(),
                "category_id": instances[i].gt_classes.item(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "keypoints": instances[i].gt_keypoints.tensor[0].tolist(),
            }
            for key in (self._column_name, "has_reap"):
                obj[key] = dataset_dict["annotations"][i][key]
            new_annos.append(obj)
        dataset_dict["annotations"] = new_annos
        return dataset_dict
