"""Evaluator for Detectron2 models.

TODO(feature): This wrapper attempts to separate framework-specific and
model-specific from the main test script.

This code is inspired by
https://github.com/yizhe-ang/detectron2-1/blob/master/detectron2_1/adv.py
"""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval
import numpy as np
import pandas as pd
import torch
from adv_patch_bench.attacks import attack_util, attacks, base_attack
from adv_patch_bench.dataloaders import reap_util
from adv_patch_bench.transforms import (
    reap_object,
    render_image,
    render_object,
    syn_object,
)
from adv_patch_bench.utils.types import (
    ImageTensor,
    ImageTensorDet,
    MaskTensor,
    SizeMM,
    SizePatch,
    SizePx,
    Target,
)
from detectron2 import structures
from detectron2.config import global_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import pairwise_iou
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

_DEFAULT_IOU_THRESHOLDS = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
)


class DetectronEvaluator:
    """Evaluator object runs evaluation for Detectron2 models."""

    def __init__(
        self,
        config_eval: Dict[str, Any],
        config_attack: Dict[str, Any],
        model: torch.nn.Module,
        dataloader: Any,
        class_names: List[str],
        all_iou_thres: np.ndarray = _DEFAULT_IOU_THRESHOLDS,
    ) -> None:
        """Evaluator wrapper for detectron model.

        Args:
            config_eval: Dictionary containing eval parameters.
            config_attack: Dictionary containing attack parameters.
            model: Target model.
            dataloader: Dataset to run attack on.
            class_names: List of class names in string.
            all_iou_thres: Array of IoU thresholds for computing score.
        """
        # General params
        self._dataset: str = config_eval["dataset"]
        self._synthetic: bool = config_eval["synthetic"]
        self._model: torch.nn.Module = model
        self._device: Any = self._model.device
        self._dataloader = dataloader
        self._input_format: str = global_cfg.INPUT.FORMAT
        self._metadata = MetadataCatalog.get(self._dataset)
        self._verbose: bool = config_eval["verbose"]
        self._debug: bool = config_eval["debug"]

        interp: str = config_eval["interp"]
        num_eval: Optional[int] = config_eval["num_eval"]
        self._img_size: SizePx = config_eval["img_size"]
        self._num_eval: Optional[int]
        self._num_eval = num_eval if num_eval is not None else len(dataloader)
        self._conf_thres: float = config_eval["conf_thres"]
        self._class_names: List[str] = class_names
        self._obj_class: int = config_eval["obj_class"]
        self._other_sign_class: int = config_eval["other_sign_class"]
        # TODO(feature): Make this an option
        self._fixed_input_size = False

        # Common keyword args for constructing RenderImage
        self._rimg_kwargs: Dict[str, Any] = {
            "img_size": self._img_size if self._fixed_input_size else None,
            "img_mode": self._input_format,
            "interp": interp,
            "img_aug_prob_geo": config_eval["img_aug_prob_geo"],
            "is_detectron": True,
        }

        # Load annotation DataFrame. "Other" signs are discarded.
        self._anno_df: pd.DataFrame = reap_util.load_annotation_df(
            config_eval["tgt_csv_filepath"]
        )
        self._annotated_signs_only: bool = config_eval["annotated_signs_only"]

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

        self._obj_size_px: SizePx = config_eval["obj_size_px"]
        self._obj_size_mm: SizeMM = config_eval["obj_size_mm"]
        self._robj_fn: render_object.RenderObject
        self._robj_kwargs: Dict[str, Any]
        robj_kwargs = {
            "obj_size_px": self._obj_size_px,
            "interp": interp,
        }
        if self._synthetic:
            self._robj_fn = syn_object.SynObject
            self._robj_kwargs = {
                **robj_kwargs,
                "obj_class": self._obj_class,
                "syn_obj_path": config_eval["syn_obj_path"],
                "syn_rotate": config_eval["syn_rotate"],
                "syn_translate": config_eval["syn_translate"],
                "syn_scale": config_eval["syn_scale"],
                "syn_3d_dist": config_eval["syn_3d_dist"],
                "syn_colorjitter": config_eval["syn_colorjitter"],
                "is_detectron": True,
            }
        else:
            self._robj_fn = reap_object.ReapObject
            self._robj_kwargs = {
                **robj_kwargs,
                "reap_transform_mode": config_eval["reap_transform_mode"],
                "reap_use_relight": config_eval["reap_use_relight"],
            }

        # Set up attack if applicable
        self._attack_type: str = config_eval["attack_type"]
        self._use_attack: bool = self._attack_type != "none"
        self._attack: Optional[base_attack.DetectorAttackModule] = None
        # Set up attack when running  "per-sign"
        if self._attack_type == "per-sign":
            self._attack = attacks.setup_attack(
                config_attack=config_attack,
                is_detectron=True,
                model=model,
                input_size=self._img_size,
                verbose=self._verbose,
            )
        self._adv_patch_path: str = config_eval["adv_patch_path"]
        self._patch_size_mm: SizePatch = config_eval["patch_size_mm"]

        # Visualization params
        self._num_vis: int = config_eval.get("num_vis", 0)
        self._vis_save_dir: pathlib.Path = (
            pathlib.Path(config_eval["result_dir"]) / "vis"
        )
        self._vis_save_dir.mkdir(exist_ok=True)
        if config_eval["vis_conf_thres"] is not None:
            self._vis_conf_thres: float = config_eval["vis_conf_thres"]
        else:
            self._vis_conf_thres: float = config_eval["conf_thres"]
        self._vis_show_bbox: bool = config_eval["vis_show_bbox"]

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

    def _log(self, *args, **kwargs) -> None:
        if self._verbose:
            print(*args, **kwargs)

    def _syn_eval(self, outputs: Target, target_render: Target):
        """Compute and save metrics for synthetic data evaluation.

        Args:
            outputs: Predicted outputs by model.
            target_render: Target of rendered image.
        """
        instances: structures.Instances = outputs["instances"]
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

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Runs evaluator and saves the prediction results.

        Returns:
            coco_instances_results: COCO-style predicted instance results.
            metrics: Dict of metrics.
        """
        coco_instances_results: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {}
        adv_patch: Optional[ImageTensor]
        patch_mask: Optional[MaskTensor]

        # Prepare attack data
        adv_patch, patch_mask = attack_util.prep_adv_patch(
            attack_type=self._attack_type,
            adv_patch_path=self._adv_patch_path,
            patch_size_mm=self._patch_size_mm,
            obj_size_px=self._obj_size_px,
            obj_size_mm=self._obj_size_mm,
        )
        if adv_patch is not None:
            adv_patch = adv_patch.to(self._device)
        if patch_mask is not None:
            patch_mask = patch_mask.to(self._device)

        total_num_images, total_num_patches, num_vis = 0, 0, 0
        eval_img_ids: List[int] = []
        self.evaluator.reset()
        self._reset_syn_metrics()

        for batch in tqdm(self._dataloader):

            if total_num_images >= self._num_eval:
                break

            file_name: str = batch[0]["file_name"]
            filename: str = file_name.split("/")[-1]
            image_id: int = batch[0]["image_id"]
            img_df: pd.DataFrame = self._anno_df[
                self._anno_df["filename"] == filename
            ]
            is_included: bool = False
            vis_name: str = filename.split(".")[0]
            obj_ids: List[int] = []

            if self._annotated_signs_only and img_df.empty:
                # Skip image if there's no annotation
                continue

            rimg: render_image.RenderImage = render_image.RenderImage(
                self._dataset,
                batch[0],
                img_df,
                **self._rimg_kwargs,
            )
            robj: render_object.RenderObject

            if self._synthetic:
                # Attacking synthetic signs
                self._log(f"Attacking {filename} ...")
                rimg.create_object(None, self._robj_fn, self._robj_kwargs)
                robj = rimg.get_object()
                robj.load_adv_patch(adv_patch=adv_patch, patch_mask=patch_mask)
                # Image is used as background only so we can include any of
                # them when evaluating synthetic signs.
                is_included = True
                total_num_patches += 1

            elif not img_df.empty:
                # Attacking real signs
                # Iterate through annotated objects in the current image
                for _, obj in img_df.iterrows():

                    obj_id: int = obj["object_id"]
                    obj_class: int = self._class_names.index(obj["final_shape"])

                    # Skip if it is "other" class or not from desired class
                    if obj_class == self._other_sign_class or (
                        obj_class != self._obj_class and self._obj_class != -1
                    ):
                        continue

                    is_included = True
                    total_num_patches += 1
                    obj_ids.append(str(obj_id))
                    if not self._use_attack:
                        continue

                    # Create RenderObject of obj_id in rimg
                    rimg.create_object(obj_id, self._robj_fn, self._robj_kwargs)

                    # TODO(feature): Should we put only one adversarial patch
                    # per image? i.e., attacking only one sign per image.
                    self._log(f"Attacking {filename} on obj {obj_id}...")

                    if self._attack_type == "per-sign":
                        # Run attack for each sign to get a new `adv_patch`
                        adv_patch: ImageTensor = self._attack.run(
                            [rimg], patch_mask.to(self._device)
                        )
                    # Load adv patch to associated RenderObject
                    robj = rimg.get_object(obj_id)
                    robj.load_adv_patch(
                        adv_patch=adv_patch, patch_mask=patch_mask
                    )

                vis_name += "_" + "-".join(obj_ids)

            if not is_included:
                # Skip image without any adversarial patch when attacking
                continue

            # Apply adversarial patch and convert to Detectron2 input format
            img_render, target_render = rimg.apply_objects()
            img_render_det: ImageTensorDet = rimg.post_process_image(img_render)

            # Perform inference on perturbed image. For REAP, COCO evaluator
            # requires ouput in original size, but synthetic object requires
            # custom evaluator so we simply use the image size.
            outputs: Dict[str, Any] = self.predict(
                img_render_det,
                rimg.img_size if self._synthetic else rimg.img_size_orig,
            )

            # Evaluate outputs and save predictions
            if self._synthetic:
                self._syn_eval(outputs, target_render)
            else:
                self.evaluator.process([target_render], [outputs])
                # Convert to coco predictions format
                instance_dicts = self._create_instance_dicts(outputs, image_id)
                coco_instances_results.extend(instance_dicts)

            # Visualization
            if num_vis < self._num_vis:
                num_vis += 1
                vis_name = f"{total_num_images}_{vis_name}"
                self._visualize(vis_name, rimg, img_render_det, target_render)

            total_num_images += 1
            eval_img_ids.append(image_id)

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
            self._log("There are no valid predictions.")
            return coco_instances_results, metrics

        metrics["bbox"]["total_num_patches"] = total_num_patches
        metrics["bbox"]["all_iou_thres"] = self._all_iou_thres.cpu().numpy()
        return coco_instances_results, metrics

    def _visualize(
        self,
        name: str,
        rimg: render_image.RenderImage,
        img_render: ImageTensorDet,
        target_render: Target,
    ) -> None:
        """Visualize ground truth, clean and adversarial predictions."""
        # Rerun prediction on rendered images to get outputs of correct size
        img_orig: ImageTensorDet = rimg.post_process_image()
        target_orig: Target = rimg.target
        output_orig: Dict[str, Any] = self.predict(img_orig, rimg.img_size)
        img_orig_np: np.ndarray = self._vis_convert_img(img_orig)

        # Visualize ground truth labels on original image
        vis_orig = Visualizer(img_orig_np, self._metadata, scale=0.5)
        if self._vis_show_bbox:
            im_gt_orig = vis_orig.draw_dataset_dict(target_orig)
        else:
            im_gt_orig = vis_orig.get_output()
        im_gt_orig.save(str(self._vis_save_dir / f"gt_orig_{name}.png"))

        # Visualize prediction on original image
        instances: structures.Instances = output_orig["instances"].to("cpu")
        # Set confidence threshold and visualize rendered image
        im_pred_orig = vis_orig.draw_instance_predictions(
            instances[instances.scores > self._vis_conf_thres]
        )
        im_pred_orig.save(str(self._vis_save_dir / f"pred_orig_{name}.png"))

        if not self._use_attack and self._dataset == "reap":
            return

        img_render_np: np.ndarray = self._vis_convert_img(img_render)
        vis_render = Visualizer(img_render_np, self._metadata, scale=0.5)

        if self._synthetic:
            # Visualize ground truth on perturbed image
            im_gt_render = vis_render.draw_dataset_dict(target_render)
            im_gt_render.save(str(self._vis_save_dir / f"gt_render_{name}.png"))

        # Visualize prediction on perturbed image
        output_render: Dict[str, Any] = self.predict(img_render, rimg.img_size)
        instances: structures.Instances = output_render["instances"].to("cpu")
        if self._vis_show_bbox:
            im_pred_render = vis_render.draw_instance_predictions(
                instances[instances.scores > self._vis_conf_thres]
            )
        else:
            im_pred_render = vis_render.get_output()
        im_pred_render.save(str(self._vis_save_dir / f"pred_render_{name}.png"))

    def _vis_convert_img(self, image: ImageTensorDet) -> np.ndarray:
        """Converge image in Detectron input format to the visualizer's."""
        if self._input_format == "BGR":
            image = image.flip(0)
        return image.permute(1, 2, 0).cpu().numpy()

    def _create_instance_dicts(
        self, outputs: Dict[str, Any], image_id: int
    ) -> List[Dict[str, Any]]:
        """Convert model outputs to coco predictions format.

        Args:
            outputs: Output dictionary from model output.
            image_id: Image ID

        Returns:
            List of per instance predictions
        """
        instance_dicts = []

        instances = outputs["instances"]
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

    def predict(
        self,
        image: ImageTensor,
        output_size: SizePx,
    ) -> Dict[str, Any]:
        """Simple inference on a single image.

        Args:
            image: Input image must be torch.Tensor of shape (C, H, W) where
                color channels are RGB (not BGR).

        Returns:
            predictions: the output of the model for one image only.
        """
        with torch.no_grad():
            # Image must be tensor with shape [3, H, W] with values [0, 255]
            inputs = {
                "image": image,
                "height": output_size[0],  # Use original height and width
                "width": output_size[1],
            }
            predictions = self._model([inputs])[0]
            return predictions
