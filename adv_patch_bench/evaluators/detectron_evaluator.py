"""Evaluator for Detectron2 models.

TODO(feature): This wrapper attempts to separate framework-specific and
model-specific from the main test script.

This code is inspired by
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/adv.py
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from detectron2 import structures
from detectron2.config import global_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval
import adv_patch_bench.utils.image as img_util
from adv_patch_bench.attacks import attack_util, base_attack
from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.transforms import render_image
from adv_patch_bench.utils.types import (
    ImageTensor,
    ImageTensorDet,
    SizeMM,
    SizePx,
    Target,
)

logger = logging.getLogger(__name__)

_DEFAULT_IOU_THRESHOLDS = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
)


class DetectronEvaluator:
    """Evaluator object runs evaluation for Detectron2 models."""

    def __init__(
        self,
        config: dict[str, dict[str, Any]],
        model: torch.nn.Module,
        dataloader: Any,
        class_names: list[str],
        all_iou_thres: np.ndarray = _DEFAULT_IOU_THRESHOLDS,
    ) -> None:
        """Evaluator wrapper for detectron model.

        Args:
            config: Dictionary containing config parameters.
            model: Target model.
            dataloader: Dataset to run attack on.
            class_names: List of class names in string.
            all_iou_thres: Array of IoU thresholds for computing score.
        """
        config_base: dict[str, str | float | int] = config["base"]
        # General params
        self._dataset: str = config_base["dataset"]
        self._synthetic: bool = config_base["synthetic"]
        self._model: torch.nn.Module = model
        self._device: Any = self._model.device
        self._dataloader = dataloader
        self._input_format: str = global_cfg.INPUT.FORMAT
        self._metadata = MetadataCatalog.get(self._dataset)

        interp: str = config_base["interp"]
        self._img_size: SizePx = SizePx(config_base["img_size"])
        num_eval: int | None = config_base["num_eval"]
        self._num_eval: int | None = (
            len(dataloader) if num_eval is None else num_eval
        )
        self._class_names: list[str] = class_names
        self._obj_class: int = config_base["obj_class"]

        # Common keyword args for constructing RenderImage
        self._rimg_kwargs: dict[str, Any] = {
            "img_mode": self._input_format,
            "interp": interp,
            "img_aug_prob_geo": config_base["img_aug_prob_geo"],
            "device": self._device,
            "obj_class": self._obj_class,
            "mode": "synthetic" if self._synthetic else "reap",
        }

        # Load annotation DataFrame. "Other" signs are discarded.
        self._anno_df: pd.DataFrame = reap_util.load_annotation_df(
            config_base["tgt_csv_filepath"]
        )
        self._annotated_signs_only: bool = config_base["annotated_signs_only"]

        # Build COCO evaluator
        self.evaluator = cocoeval.CustomCOCOEvaluator(
            self._dataset,
            ["bbox"],
            False,
            output_dir=global_cfg.OUTPUT_DIR,
            use_fast_impl=False,
        )

        # Set up list of IoU thresholds to consider
        self._all_iou_thres = torch.from_numpy(all_iou_thres).to(self._device)

        self._obj_size_px: SizePx = SizePx(
            config_base.get("obj_size_px", (0, 0))
        )
        self._obj_size_mm: SizeMM = SizePx(
            config_base.get("obj_size_mm", (0.0, 0.0))
        )
        robj_kwargs = {
            "dataset": self._dataset,
            "obj_size_px": self._obj_size_px,
            "interp": interp,
        }
        keyword = "syn" if self._synthetic else "reap"
        self._robj_kwargs: dict[str, Any] = {
            **robj_kwargs,
            **{k: v for k, v in config_base.items() if keyword in k},
        }

        # Set up attack if applicable
        self._attack_type: str = config_base["attack_type"]
        self._use_attack: bool = self._attack_type != "none"
        self._attack: base_attack.DetectorAttackModule | None = None
        # Set up attack when running  "per-sign"
        if self._attack_type == "per-sign":
            self._attack = attack_util.setup_attack(config=config, model=model)
        self._adv_patch_path: str = config_base["adv_patch_path"]
        self._patch_size = config_base.get("patch_size")

        # Visualization params
        self._num_vis: int = config_base.get("num_vis", 0)
        self._vis_save_dir: pathlib.Path = (
            pathlib.Path(config_base["result_dir"]) / "vis"
        )
        self._vis_save_dir.mkdir(exist_ok=True)
        if config_base["vis_conf_thres"] is not None:
            self._vis_conf_thres: float = config_base["vis_conf_thres"]
        else:
            self._vis_conf_thres: float = config_base["conf_thres"]
        self._vis_show_bbox: bool = config_base["vis_show_bbox"]
        self._vis_img_scale: float = 0.5

        # Variables for storing synthetic data results
        # syn_scores and syn_matches have shape [num_ious, num_eval]
        self.syn_scores: torch.Tensor
        self.syn_matches: torch.Tensor
        self._syn_idx: int  # Keep track of next index of sample to update

    def _reset_syn_metrics(self):
        # Tensors for saving metrics for synthetic benchmark
        self.syn_scores = torch.zeros(
            (len(self._all_iou_thres), self._num_eval), device=self._device
        )
        self.syn_matches = torch.zeros_like(self.syn_scores)
        self._syn_idx = 0

    def _syn_eval(self, outputs: list[Target], target_renders: list[Target]):
        """Compute and save metrics for synthetic data evaluation.

        Args:
            outputs: Predicted outputs by model.
            target_renders: Target of rendered image.
        """
        for output, target_render in zip(outputs, target_renders):
            instances: structures.Instances = output["instances"]
            # Filter both dt by class
            instances = instances[instances.pred_classes == self._obj_class]
            # Last gt box is the new one we added for synthetic sign
            # Output of pairwise_iou here has shape [num_dts, 1]
            ious: torch.Tensor = pairwise_iou(
                instances.pred_boxes,
                target_render["instances"].gt_boxes[-1].to(self._device),
            )[:, 0]

            # Skip empty ious (no overlap)
            if len(ious) == 0:
                self._syn_idx += 1
                return

            # Find the match with highest IoU. This matches COCO evaluator.
            max_iou, max_idx = ious.max(0, keepdim=True)
            matches = max_iou >= self._all_iou_thres
            # Zero out scores if there's no match (i.e., IoU lower than threshold)
            scores = instances.scores[max_idx] * matches
            # Save scores and gt-dt matches at each level of IoU thresholds
            self.syn_matches[:, self._syn_idx] = matches
            self.syn_scores[:, self._syn_idx] = scores
            self._syn_idx += 1

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Runs evaluator and saves the prediction results.

        Returns:
            coco_instances_results: COCO-style predicted instance results.
            metrics: Dict of metrics.
        """
        coco_instances_results: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        # Prepare attack data
        adv_patch, patch_mask = attack_util.prep_adv_patch(
            attack_type=self._attack_type,
            adv_patch_path=self._adv_patch_path,
            patch_size=self._patch_size,
            obj_size_px=self._obj_size_px,
            obj_size_mm=self._obj_size_mm,
        )
        if adv_patch is not None:
            img_util.coerce_rank(adv_patch, 4)
            adv_patch = adv_patch.to(self._device)
        if patch_mask is not None:
            img_util.coerce_rank(patch_mask, 4)
            patch_mask = patch_mask.to(self._device)

        total_num_images, total_num_patches, num_vis = 0, 0, 0
        eval_img_ids: list[int] = []
        self.evaluator.reset()
        self._reset_syn_metrics()

        for batch in tqdm(self._dataloader):

            if total_num_images >= self._num_eval:
                break

            filenames: list[str] = [
                b["file_name"].split("/")[-1] for b in batch
            ]
            image_ids: list[int] = [b["image_id"] for b in batch]
            filenames = [name.split(".")[0] for name in filenames]

            rimg: render_image.RenderImage = render_image.RenderImage(
                samples=batch,
                robj_kwargs=self._robj_kwargs,
                **self._rimg_kwargs,
            )

            if self._synthetic:
                num_objs = len(rimg.images)
            elif rimg.num_objs > 0:
                num_objs = rimg.num_objs
            else:
                # Skip image without any adversarial patch when attacking
                continue

            cur_adv_patch, cur_patch_mask = None, None
            total_num_patches += num_objs
            # Repeat patch and mask for each object in batch of images
            repeat_size = (num_objs, 1, 1, 1)
            if patch_mask is not None:
                cur_patch_mask = patch_mask.repeat(repeat_size)
            if self._attack_type == "per-sign":
                cur_adv_patch = self._attack.run(
                    rimg,
                    cur_patch_mask,
                    batch_mode=True,
                )
            elif adv_patch is not None:
                cur_adv_patch = adv_patch.repeat(repeat_size)

            # Apply adversarial patch and convert to Detectron2 input format
            img_render, target_render = rimg.apply_objects(
                cur_adv_patch, cur_patch_mask
            )
            img_render_det: ImageTensorDet = rimg.post_process_image(img_render)

            # Perform inference on perturbed image. For REAP, COCO evaluator
            # requires ouput in original size, but synthetic object requires
            # custom evaluator so we simply use the image size.
            outputs: list[dict[str, Any]] = self.predict(img_render_det)

            # Evaluate outputs and save predictions
            if self._synthetic:
                self._syn_eval(outputs, target_render)
            else:
                self.evaluator.process(target_render, outputs)
                # Convert to coco predictions format
                instance_dicts = self._create_instance_dicts(outputs, image_ids)
                coco_instances_results.extend(instance_dicts)

            # Visualization
            if num_vis < self._num_vis:
                num_vis += len(outputs)
                vis_names = [
                    f"{total_num_images + i}_{name}"
                    for i, name in enumerate(filenames)
                ]
                self._visualize(vis_names, rimg, img_render_det, target_render)

            total_num_images += len(outputs)
            eval_img_ids.extend(image_ids)

        # Compute final metrics
        if self._synthetic:
            metrics = {
                "bbox": {
                    "syn_scores": self.syn_scores.cpu().numpy(),
                    "syn_matches": self.syn_matches.float().cpu().numpy(),
                }
            }
        else:
            metrics = self.evaluator.evaluate(img_ids=eval_img_ids)
        if "bbox" not in metrics:
            logger.info("There are no valid predictions.")
            return coco_instances_results, metrics

        metrics["bbox"]["total_num_patches"] = total_num_patches
        metrics["bbox"]["all_iou_thres"] = self._all_iou_thres.cpu().numpy()
        return coco_instances_results, metrics

    def _visualize(
        self,
        file_names: list[str],
        rimg: render_image.RenderImage,
        imgs_render: ImageTensorDet,
        targets_render: list[Target],
    ) -> None:
        """Visualize ground truth, clean and adversarial predictions."""
        # Rerun prediction on rendered images to get outputs of correct size
        imgs_orig: ImageTensorDet = rimg.post_process_image()
        targets_orig: list[Target] = rimg.samples
        outputs_orig: list[dict[str, Any]] = self.predict(imgs_orig)
        imgs_orig_np: np.ndarray = self._vis_convert_img(imgs_orig)

        for img_orig_np, target_orig, output_orig, name in zip(
            imgs_orig_np, targets_orig, outputs_orig, file_names
        ):
            # Visualize ground truth labels on original image
            vis_gt_orig = Visualizer(
                img_orig_np, self._metadata, scale=self._vis_img_scale
            )
            if self._vis_show_bbox:
                im_gt_orig = vis_gt_orig.draw_dataset_dict(target_orig)
            else:
                im_gt_orig = vis_gt_orig.get_output()
            im_gt_orig.save(str(self._vis_save_dir / f"gt_orig_{name}.png"))

            # Visualize prediction on original image
            vis_pred_orig = Visualizer(
                img_orig_np, self._metadata, scale=self._vis_img_scale
            )
            instances: structures.Instances = output_orig["instances"].to("cpu")
            # Set confidence threshold and visualize rendered image
            im_pred_orig = vis_pred_orig.draw_instance_predictions(
                instances[instances.scores > self._vis_conf_thres]
            )
            im_pred_orig.save(str(self._vis_save_dir / f"pred_orig_{name}.png"))

        if not self._use_attack and self._dataset == "reap":
            return

        imgs_render_np: np.ndarray = self._vis_convert_img(imgs_render)
        outputs_render: list[dict[str, Any]] = self.predict(imgs_render)

        for img_render_np, target_render, output_render, name in zip(
            imgs_render_np, targets_render, outputs_render, file_names
        ):

            if self._synthetic:
                # Visualize ground truth on perturbed image
                vis_gt_render = Visualizer(
                    img_render_np, self._metadata, scale=self._vis_img_scale
                )
                im_gt_render = vis_gt_render.draw_dataset_dict(target_render)
                im_gt_render.save(
                    str(self._vis_save_dir / f"gt_render_{name}.png")
                )

            # Visualize prediction on perturbed image
            vis_pred_render = Visualizer(
                img_render_np, self._metadata, scale=self._vis_img_scale
            )
            instances: structures.Instances = output_render["instances"].to(
                "cpu"
            )
            if self._vis_show_bbox:
                im_pred_render = vis_pred_render.draw_instance_predictions(
                    instances[instances.scores > self._vis_conf_thres]
                )
            else:
                im_pred_render = vis_pred_render.get_output()
            im_pred_render.save(
                str(self._vis_save_dir / f"pred_render_{name}.png")
            )

    def _vis_convert_img(self, image: ImageTensorDet) -> np.ndarray:
        """Converge image in Detectron input format to the visualizer's."""
        if self._input_format == "BGR":
            image = image.flip(1)
        return image.permute(0, 2, 3, 1).cpu().numpy()

    def _create_instance_dicts(
        self, outputs: list[dict[str, Any]], image_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Convert model outputs to coco predictions format.

        Args:
            outputs: Output dictionary from model output.
            image_id: Image ID

        Returns:
            List of per instance predictions
        """
        instance_dicts = []

        for output, image_id in zip(outputs, image_ids):
            instances = output["instances"]
            pred_classes = instances.pred_classes.cpu().numpy()
            pred_boxes = instances.pred_boxes
            scores = instances.scores.cpu().numpy()

            # For each bounding box
            for i, box in enumerate(pred_boxes):
                i_dict = {
                    "image_id": image_id,
                    "category_id": int(pred_classes[i]),
                    "bbox": [b.item() for b in box],
                    "score": float(scores[i]),
                }
                instance_dicts.append(i_dict)

        return instance_dicts

    @torch.no_grad()
    def predict(
        self,
        images: ImageTensor,
        output_size: SizePx | None = None,
    ) -> dict[str, Any]:
        """Simple inference on a single image.

        Args:
            image: Input image must be torch.Tensor of shape (C, H, W) where
                color channels are RGB (not BGR).

        Returns:
            predictions: the output of the model for one image only.
        """
        if output_size is None:
            output_size = images.shape[-2:]
        # Image must be tensor with shape [B, 3, H, W] with values [0, 255]
        inputs = []
        for image in images:
            inputs.append(
                {
                    "image": image,
                    "height": output_size[0],  # Use original height and width
                    "width": output_size[1],
                }
            )
        predictions = self._model(inputs)
        return predictions
