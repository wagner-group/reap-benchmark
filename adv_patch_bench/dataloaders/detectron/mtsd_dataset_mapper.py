"""Registers datasets, and defines other dataloading utilities."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import detectron2
import numpy as np
import torch
import torchvision
from detectron2.config import global_cfg
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.dataloaders.detectron import reap_dataset_mapper
from adv_patch_bench.transforms import lighting_tf, util
from adv_patch_bench.utils.types import DetectronSample
from hparams import DATASET_METADATA, DEFAULT_SYN_OBJ_DIR

logger = logging.getLogger(__name__)


class MtsdDatasetMapper(reap_dataset_mapper.ReapDatasetMapper):
    """A callable which takes a dataset dict in Detectron2 Dataset format.

    This is the default callable to be used to map your dataset dict into
    training data.
    """

    def __init__(
        self,
        cfg: detectron2.config.CfgNode,
        config_base: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initialize MtsdDatasetMapper. See ReapDatasetMapper for args.

        Args:
            cfg: Detectron2 config.
            config_base: Base config.
        """
        super().__init__(cfg, **kwargs)
        metadata = DATASET_METADATA["mtsd_no_color"]
        class_names = metadata["class_name"]
        hw_ratio_dict = metadata["hw_ratio"]
        shape_dict = metadata["shape"]

        self._syn_objs: dict[int, torch.Tensor] = {}
        self._syn_obj_masks: dict[int, torch.Tensor] = {}
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

    def __call__(self, dataset_dict: DetectronSample):
        """Modify sample directly loaded from Detectron2 dataset.

        Args:
            dataset_dict: Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        column_name = f'{self._relight_params["method"]}_coeffs'
        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )
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
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens,
                image,
            )
        else:
            # Scale and shift bbox according to new image size
            for anno in dataset_dict["annotations"]:
                xmin, ymin, xmax, ymax = anno["bbox"]
                ymin = ymin * scales[0] + padding[1]
                ymax = ymax * scales[0] + padding[1]
                xmin = xmin * scales[1] + padding[0]
                xmax = xmax * scales[1] + padding[0]
                anno["bbox"] = [xmin, ymin, xmax, ymax]

            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of
        # pickle & mp.Queue. Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = image

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)
                get_mtsd_transforms(
                    anno,
                    image,
                    self._syn_objs,
                    self._syn_obj_masks,
                    self._relight_params,
                    bg_class=global_cfg.other_catId,
                )

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

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
            for key in (column_name, "has_reap"):
                obj[key] = dataset_dict["annotations"][i][key]
            new_annos.append(obj)
        dataset_dict["annotations"] = new_annos

        return dataset_dict


def get_mtsd_transforms(
    anno: dict[str, Any],
    image: torch.Tensor | np.ndarray,
    syn_objs: dict[int, torch.Tensor],
    syn_obj_masks: dict[int, torch.Tensor],
    relight_params: dict[str, Any],
    bg_class: int = -1,
) -> None:
    """Generate transform params for MTSD dataset.

    Args:
        anno: Annotation dict.
        image: Input image.
        syn_objs: Dict of synthetic objects.
        syn_obj_masks: Dict of synthetic object masks.
        relight_params: Relighting parameters.
        bg_class: Background class id.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    column_name = f'{relight_params["method"]}_coeffs'
    anno[column_name] = torch.tensor([[1, 0]], dtype=torch.float32)
    xmin, ymin, xmax, ymax = [int(max(0, b)) for b in anno["bbox"]]
    if xmax <= xmin or ymax <= ymin:
        logger.warning(
            "Invalid bbox: xmin, ymin, xmax, ymax = %s\nSetting category_id to "
            "background class and skipping.",
            str(anno["bbox"]),
        )
        anno["keypoints"] = np.zeros((4, 3), dtype=np.float32)
        anno["category_id"] = bg_class
        return

    anno["keypoints"] = np.array(
        [
            [xmin, ymin, 2],
            [xmax, ymin, 2],
            [xmax, ymax, 2],
            [xmin, ymax, 2],
        ],
        dtype=np.float32,
    )

    obj_class = anno["category_id"]
    if obj_class not in syn_objs:
        return

    # Compute relighting params from cropped object
    # We don't know true keypoints for MTSD objects so the mask is
    # simply scaled to match the object size  without any other
    # geometric transformations
    obj_mask = img_util.resize_and_pad(
        obj=syn_obj_masks[obj_class],
        resize_size=(ymax - ymin, xmax - xmin),
        keep_aspect_ratio=False,
        is_binary=True,
    )
    if "percentile" not in relight_params["method"]:
        relight_params["syn_obj"] = syn_objs[obj_class]
    obj = image[:, ymin:ymax, xmin:xmax]
    coeffs = lighting_tf.compute_relight_params(
        obj / 255, obj_mask=obj_mask, **relight_params
    )
    anno[column_name] = coeffs
