"""RP2 attack for Detectron2 models."""

import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from adv_patch_bench.attacks.rp2 import rp2_base
from adv_patch_bench.utils.types import ImageTensor, Target
from detectron2 import structures


class RP2AttackDetectron(rp2_base.RP2AttackModule):
    """RP2 Attack for Detectron2 models."""

    def __init__(self, attack_config, core_model, **kwargs):
        """Initialize RP2AttackDetectron.

        TODO(feature): Currently, we assume that Detectron2 models are
        Faster R-CNN so loss function and params are specific to Faster R-CNN.
        We should implement attack on Faster R-CNN as subclass of Detectron2.

        Args:
            attack_config: Dictionary of attack params.
            core_model: Traget model to attack.
        """
        super().__init__(attack_config, core_model, **kwargs)

        detectron_config: Dict[str, Any] = attack_config["detectron"]
        self._detectron_obj_const: float = detectron_config["obj_loss_const"]
        self._detectron_iou_thres: float = detectron_config["iou_thres"]

        self._nms_thresh_orig = copy.deepcopy(
            core_model.proposal_generator.nms_thresh
        )
        self._post_nms_topk_orig = copy.deepcopy(
            core_model.proposal_generator.post_nms_topk
        )
        # self.nms_thresh = 0.9
        # self.post_nms_topk = {True: 5000, False: 5000}
        self.nms_thresh = self._nms_thresh_orig
        self.post_nms_topk = self._post_nms_topk_orig

    def _on_enter_attack(self, **kwargs):
        self.is_training = self._core_model.training
        self._core_model.eval()
        self._core_model.proposal_generator.nms_thresh = self.nms_thresh
        self._core_model.proposal_generator.post_nms_topk = self.post_nms_topk

    def _on_exit_attack(self, **kwargs):
        self._core_model.train(self.is_training)
        self._core_model.proposal_generator.nms_thresh = self._nms_thresh_orig
        self._core_model.proposal_generator.post_nms_topk = (
            self._post_nms_topk_orig
        )

    def _loss_func(
        self,
        adv_img: ImageTensor,
        adv_target: Target,
        obj_class: int,
    ) -> torch.Tensor:
        """Compute loss for Faster R-CNN models.

        Args:
            adv_img: Image to compute loss on.
            adv_target: Target label to compute loss on.
            obj_class: Target object class. Usually ground-truth label for
                untargeted attack, and target class for targeted attack.

        Returns:
            Loss for attacker to minimize.
        """
        # NOTE: IoU threshold for ROI is 0.5 and for RPN is 0.7
        inputs = adv_target
        inputs["image"] = adv_img
        _, target_labels, target_logits, obj_logits = _get_targets(
            self._core_model,
            [inputs],
            device=self._core_model.device,
            iou_thres=self._detectron_iou_thres,
            score_thres=self.min_conf,
            use_correct_only=False,
        )

        # DEBUG
        # import cv2
        # from detectron2.utils.visualizer import Visualizer
        # from detectron2.data import MetadataCatalog
        # with torch.no_grad():
        #     idx = 0
        #     metadata[idx]['height'], metadata[idx]['width'] = adv_img.shape[2:]
        #     outputs = self.core_model(metadata)[idx]
        #     instances = outputs["instances"]
        #     mask = instances.scores > 0.5
        #     instances = instances[mask]
        #     self.metadata = MetadataCatalog.get('mapillary_combined')
        #     img = metadata[idx]['image'].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
        #     v = Visualizer(img, self.metadata, scale=0.5)
        #     vis_og = v.draw_instance_predictions(instances.to('cpu')).get_image()
        #     cv2.imwrite('temp_pred.png', vis_og[:, :, ::-1])
        #     metadata[idx]['annotations'] = [{
        #         'bbox': metadata[idx]['instances'].gt_boxes.tensor[0].tolist(),
        #         'category_id': metadata[idx]['instances'].gt_classes.item(),
        #         'bbox_mode': metadata[idx]['annotations'][0]['bbox_mode'],
        #     }]
        #     vis_gt = v.draw_dataset_dict(metadata[0]).get_image()
        #     cv2.imwrite('temp_gt.png', vis_gt[:, :, ::-1])
        #     print('ok')
        # import pdb
        # pdb.set_trace()

        # Loop through each EoT image
        loss: torch.Tensor = torch.zeros(1, device=adv_img.device)
        for tgt_lb, tgt_log, obj_log in zip(
            target_labels, target_logits, obj_logits
        ):
            # Filter obj_class
            if "obj_class_only" in self.attack_mode:
                # Focus attack on prediction of `obj_class` only
                idx = obj_class == tgt_lb
                tgt_lb, tgt_log, obj_log = (
                    tgt_lb[idx],
                    tgt_log[idx],
                    obj_log[idx],
                )
            else:
                tgt_lb = torch.zeros_like(tgt_lb) + obj_class
            # If there's no matched gt/prediction, then attack already succeeds.
            # TODO(feature): Appearing or misclassification attacks
            target_loss: torch.Tensor = torch.zeros_like(loss)
            obj_loss: torch.Tensor = torch.zeros_like(loss)
            if len(tgt_log) > 0 and len(tgt_lb) > 0:
                # Ignore the background class on tgt_log
                target_loss = F.cross_entropy(tgt_log, tgt_lb, reduction="mean")
            if len(obj_logits) > 0 and self._detectron_obj_const != 0:
                obj_lb = torch.ones_like(obj_log)
                obj_loss = F.binary_cross_entropy_with_logits(
                    obj_log, obj_lb, reduction="mean"
                )
            loss += target_loss + self._detectron_obj_const * obj_loss
        return -loss


def _get_targets(
    model: torch.nn.Module,
    inputs: List[Dict[str, Any]],
    device: str = "cuda",
    iou_thres: float = 0.1,
    score_thres: float = 0.1,
    use_correct_only: bool = False,
) -> Tuple[structures.Boxes, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a set of targets to attack.

    Args:
        inputs: A list containing a single dataset_dict, transformed by
            a DatasetMapper.
        iou_thres: IoU threshold for matching predicted and ground-truth
            bouing boxes.
        score_thres: Predictions with class score less than score_thres are
            dropped.

    Returns:
        target_boxes, target_labels
    """
    images = model.preprocess_image(inputs)

    # Get features
    features = model.backbone(images.tensor)

    # Get bounding box proposals. For API, see
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/proposal_generator/rpn.py#L431
    proposals, _ = model.proposal_generator(images, features, None)
    proposal_boxes = [x.proposal_boxes for x in proposals]

    # Get proposal boxes' classification scores
    predictions = _get_roi_heads_predictions(model, features, proposal_boxes)
    # predictions = get_roi_heads_predictions(model, features, proposals)

    # Scores (softmaxed) for a single image, [n_proposals, n_classes + 1]
    # scores = model.roi_heads.box_predictor.predict_probs(predictions, proposals)[0]
    # Instead, we want to get logit scores without softmax. For API, see
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L547
    class_logits, _ = predictions
    num_inst_per_image = [len(p) for p in proposals]
    # NOTE: class_logits dim [[1000, num_classes + 1], ...]
    class_logits = class_logits.split(num_inst_per_image, dim=0)

    gt_boxes = [i["instances"].gt_boxes.to(device) for i in inputs]
    gt_classes = [i["instances"].gt_classes for i in inputs]
    objectness_logits = [x.objectness_logits for x in proposals]

    outputs = _filter_positive_proposals(
        proposal_boxes,
        class_logits,
        gt_boxes,
        gt_classes,
        device=device,
        iou_thres=iou_thres,
        score_thres=score_thres,
        use_correct_only=use_correct_only,
    )
    outputs.append(class_logits)
    outputs.append(objectness_logits)
    return outputs


def _get_roi_heads_predictions(
    model,
    features: Dict[str, torch.Tensor],
    proposal_boxes: List[structures.Boxes],
    # proposals,
) -> Tuple[torch.Tensor, torch.Tensor]:
    roi_heads = model.roi_heads
    features = [features[f] for f in roi_heads.box_in_features]
    # Defn: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/poolers.py#L205
    # Usage: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/roi_heads.py#L780
    box_features = roi_heads.box_pooler(features, proposal_boxes)
    box_features = roi_heads.box_head(box_features)
    logits, proposal_deltas = roi_heads.box_predictor(box_features)
    del box_features
    # proposal_boxes = [x.proposal_boxes for x in proposals]
    # predictions = roi_heads.box_predictor(box_features)
    # pred_instances, temp = roi_heads.box_predictor.inference(predictions, proposals)
    # print('get_roi_heads_predictions')
    # import pdb
    # pdb.set_trace()
    return logits, proposal_deltas


def _filter_positive_proposals(
    proposal_boxes: List[structures.Boxes],
    class_logits: List[torch.Tensor],
    gt_boxes: List[structures.Boxes],
    gt_classes: List[torch.Tensor],
    **kwargs,
) -> List[List[Any]]:
    """See _filter_positive_proposals_single()."""
    outputs = [[], []]
    for inpt in zip(
        proposal_boxes, class_logits, gt_boxes, gt_classes
    ):
        out = _filter_positive_proposals_single(*inpt, **kwargs)
        for i, o in enumerate(out):
            outputs[i].append(o)
    return outputs


def _filter_positive_proposals_single(
    proposal_boxes: structures.Boxes,
    class_logits: torch.Tensor,
    gt_boxes: structures.Boxes,
    gt_classes: torch.Tensor,
    device: str = "cuda",
    iou_thres: float = 0.1,
    score_thres: float = 0.1,
    use_correct_only: bool = False,
) -> Tuple[structures.Boxes, torch.Tensor]:
    """Filter for desired targets for the attack.

    Args:
        proposal_boxes: Proposal boxes directly from RPN.
        scores: Softmaxed scores for each proposal box.
        gt_boxes: Ground truth boxes.
        gt_classes: Ground truth classes.

    Returns:
        Filtered target boxes and corresponding class labels.
    """
    n_proposals: int = len(proposal_boxes)

    proposal_gt_ious: torch.Tensor = structures.pairwise_iou(
        proposal_boxes, gt_boxes
    )

    # Pair each proposed box in proposal_boxes with a ground-truth box in
    # gt_boxes, i.e., find ground-truth box with highest IoU.
    # IoU with paired gt_box, idx of paired gt_box
    paired_ious, paired_gt_idx = proposal_gt_ious.max(dim=1)

    # Filter for IoUs > iou_thres
    iou_cond = paired_ious >= iou_thres

    # Get class of paired gt_box
    gt_classes_repeat = gt_classes.repeat(n_proposals, 1)
    idx = torch.arange(n_proposals)
    paired_gt_classes = gt_classes_repeat[idx, paired_gt_idx]

    if use_correct_only:
        # Filter for score of proposal > score_thres
        # Get scores of corresponding class
        scores = F.softmax(class_logits, dim=-1)
        paired_scores = scores[idx, paired_gt_classes]
        score_cond = paired_scores >= score_thres
        # Filter for positive proposals and their corresponding gt labels
        cond = iou_cond & score_cond
    else:
        cond = iou_cond

    return (proposal_boxes[cond], paired_gt_classes[cond].to(device))
