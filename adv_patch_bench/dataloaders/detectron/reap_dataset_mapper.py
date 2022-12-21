"""Registers datasets, and defines other dataloading utilities.

Code is adapted from
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/datasets.py
"""

from __future__ import annotations

import copy
import logging

import detectron2
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.utils.types import SizePx


def _build_transform_gen(cfg: detectron2.config.CfgNode, is_train: bool):
    """Create a list of :class:`TransformGen` from config.

    Now it includes only resizing.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), f"more than 2 ({len(min_size)}) min_size(s) are provided for ranges"

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    # Remove orizontal flipping
    # if is_train:
    #     tfm_gens.append(T.RandomFlip())
    logger.info("TransformGens: %s", str(tfm_gens))
    return tfm_gens


class ReapDatasetMapper:
    """A callable which takes a dataset dict in Detectron2 Dataset format.

    This is the default callable to be used to map your dataset dict into
    training data.
    """

    def __init__(
        self,
        cfg: detectron2.config.CfgNode,
        is_train: bool = True,
        img_size: SizePx | None = None,
    ) -> None:
        """Initialize REAP data mapper.

        Args:
            cfg: Detectron2 config.
            is_train: Whether we are training. Defaults to True.
            img_size: Resize all images to this fixed size instead of just
                loading the original size. Defaults to None.
        """
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(
                cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE
            )
            logging.getLogger(__name__).info(
                "CropGen used in training: %s", str(self.crop_gen)
            )
        else:
            self.crop_gen = None

        self.tfm_gens = _build_transform_gen(cfg, is_train)

        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        # Set keypoint_on to True to handle REAP geometric transform
        # self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.keypoint_on = True
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self._img_size: SizePx = img_size

    def __call__(self, dataset_dict):
        """Modify sample directly loaded from Detectron2 dataset.

        Args:
            dataset_dict: Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )
        # Convert to torch.Tensor to resize/pad and convert back to numpy
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image = img_util.resize_and_pad(
            obj=image,
            resize_size=self._img_size,
            pad_size=self._img_size,
            keep_aspect_ratio=True,
        )
        image = image.permute(1, 2, 0).numpy()
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens,
                image,
            )
        else:
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
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

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
            for key in ("alpha", "beta", "has_reap"):
                obj[key] = dataset_dict["annotations"][i][key]
            new_annos.append(obj)
        dataset_dict["annotations"] = new_annos
        return dataset_dict
