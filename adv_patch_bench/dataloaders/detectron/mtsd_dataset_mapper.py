"""Registers datasets, and defines other dataloading utilities."""

from __future__ import annotations

import copy

import detectron2
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

import adv_patch_bench.utils.image as img_util
from adv_patch_bench.dataloaders.detectron import reap_dataset_mapper
from adv_patch_bench.transforms.lighting_tf import compute_relight_params
from adv_patch_bench.utils.types import DetectronSample


class MtsdDatasetMapper(reap_dataset_mapper.ReapDatasetMapper):
    """A callable which takes a dataset dict in Detectron2 Dataset format.

    This is the default callable to be used to map your dataset dict into
    training data.
    """

    def __init__(
        self,
        cfg: detectron2.config.CfgNode,
        relight_method: str = "percentile",
        relight_percentile: float = 10.0,
        **kwargs,
    ) -> None:
        """Initialize MtsdDatasetMapper. See ReapDatasetMapper for args.

        Args:
            cfg: Detectron2 config.
            relight_method: Method for computing relighting params. Defaults to
                "percentile".
            relight_percentile: Percentile of pixels to use as min/max of the
                scaling range. See transforms.lighting_tf._find_min_max() for
                more detail. Defaults to 10.0.
        """
        super().__init__(cfg, **kwargs)
        self._relight_method: str = relight_method
        self._relight_percentile: float = relight_percentile

    def __call__(self, dataset_dict: DetectronSample):
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

                xmin, ymin, xmax, ymax = [int(max(0, b)) for b in anno["bbox"]]
                obj = image[:, ymin:ymax, xmin:xmax]
                # Compute relighting params from cropped object
                alpha, beta = compute_relight_params(
                    obj / 255,
                    method=self._relight_method,
                    percentile=self._relight_percentile,
                )
                anno["alpha"] = alpha if 0 <= alpha <= 1 else 1.0
                anno["beta"] = beta if -1 <= beta <= 1 else 0.0
                anno["keypoints"] = np.array(
                    [
                        [xmin, ymin, 2],
                        [xmax, ymin, 2],
                        [xmax, ymax, 2],
                        [xmin, ymax, 2],
                    ],
                    dtype=np.float32,
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
            for key in ("alpha", "beta", "has_reap"):
                obj[key] = dataset_dict["annotations"][i][key]
            new_annos.append(obj)
        dataset_dict["annotations"] = new_annos

        return dataset_dict
