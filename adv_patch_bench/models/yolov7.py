"""Implementation of YOLOv7.

Adapted from
https://github.com/jinfagang/yolov7_d2/blob/main/yolov7/modeling/meta_arch/yolov7.py
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from alfred.dl.metrics.iou_loss import ciou
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm
from detectron2.utils.logger import log_first_n
from torch import nn
from yolov7.modeling.neck.yolo_fpn import YOLOFPN
from yolov7.modeling.neck.yolo_pafpn import YOLOPAFPN
from yolov7.utils.boxes import bboxes_iou

from adv_patch_bench.models.custom_build import CUSTOM_META_ARCH_REGISTRY

__all__ = ["YOLOV7", "YOLOHead"]
supported_backbones = [
    "resnet",
    "res2net",
    "regnet",
    "swin",
    "efficient",
    "darknet",
    "pvt",
]

logger = logging.getLogger(__name__)


@CUSTOM_META_ARCH_REGISTRY.register()
class YOLOV7(nn.Module):
    """YOLO model. Darknet 53 is the default backbone of this model."""

    def __init__(self, cfg):
        """Initialize YOLOV7.

        Args:
            cfg: Detectron2 config.
        """
        super().__init__()
        # configurations
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.conf_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.nms_threshold = cfg.MODEL.YOLO.NMS_THRESHOLD
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.neck_type = cfg.MODEL.YOLO.NECK.TYPE
        self.with_spp = cfg.MODEL.YOLO.NECK.WITH_SPP
        self.depth_mul = cfg.MODEL.YOLO.DEPTH_MUL
        self.width_mul = cfg.MODEL.YOLO.WIDTH_MUL

        self.max_iter = cfg.SOLVER.MAX_ITER
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.max_boxes_num = cfg.MODEL.YOLO.MAX_BOXES_NUM
        self.in_features = cfg.MODEL.YOLO.IN_FEATURES

        self.change_iter = 10
        self.iter = 0
        self.use_l1 = False
        anchors = cfg.MODEL.YOLO.ANCHORS

        logger.info(
            "YOLOv7 params: num_classes=%s, max_boxes_num=%s, in_features=%s, "
            "conf_threshold=%s, nms_threshold=%s, anchors=%s",
            str(self.num_classes),
            str(self.max_boxes_num),
            str(self.in_features),
            str(self.conf_threshold),
            str(self.nms_threshold),
            str(anchors),
        )

        assert (
            len(
                [i for i in supported_backbones if i in cfg.MODEL.BACKBONE.NAME]
            )
            > 0
        ), f"Only {supported_backbones} supported."

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.size_divisibility = (
            32
            if self.backbone.size_divisibility == 0
            else self.backbone.size_divisibility
        )
        backbone_shape = [backbone_shape[i].channels for i in self.in_features]

        if comm.is_main_process():
            logger.info("YOLO.ANCHORS: %s", str(anchors))
            logger.info("backboneshape: %s", str(backbone_shape))

        if self.neck_type == "pafpn":
            width_mul = backbone_shape[0] / 256
            self.neck = YOLOPAFPN(
                depth=self.depth_mul,
                width=width_mul,
                in_features=self.in_features,
            )
            neck_dims = backbone_shape
        else:
            self.neck = YOLOFPN(
                width=self.width_mul,
                in_channels=backbone_shape,
                in_features=self.in_features,
                with_spp=self.with_spp,
            )
            # 256, 512, 1024 -> 1024, 512, 256
            neck_dims = self.neck.out_channels

        self.convs_list = nn.ModuleList(
            nn.Conv2d(x, len(anchors[0]) * (5 + cfg.MODEL.YOLO.CLASSES), 1)
            for x in neck_dims
        )

        self.register_buffer(
            "pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        )
        self.normalizer = (
            lambda x: (x / 255.0 - self.pixel_mean) / self.pixel_std
        )
        self.padded_value = cfg.MODEL.PADDED_VALUE / 255.0
        self.loss_evaluators = [
            YOLOHead(cfg, anchor, level) for level, anchor in enumerate(anchors)
        ]
        self.to(self.device)

    def update_iter(self, i):
        """Update iteration."""
        self.iter = i

    def _make_cbl(self, _in, _out, ks):
        """Missing documentations.

        TODO(documentation): cbl = conv + batch_norm + leaky_relu.
        """
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            _in,
                            _out,
                            kernel_size=ks,
                            stride=1,
                            padding=pad,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(_out)),
                    ("relu", nn.LeakyReLU(0.1)),
                ]
            )
        )

    def _make_embedding(self, filters_list, in_filters, out_filter):
        modules_list = nn.ModuleList(
            [
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
            ]
        )
        modules_list.add_module(
            "conv_out",
            nn.Conv2d(
                filters_list[1],
                out_filter,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        return modules_list

    def preprocess_image(self, batched_inputs, training: bool):
        """Preprocess image."""
        images = [x["image"].to(self.device) for x in batched_inputs]
        batch_size = len(images)
        images = [self.normalizer(x) for x in images]

        images = ImageList.from_tensors(
            images,
            size_divisibility=self.size_divisibility,
            pad_value=self.padded_value,
        )

        # sync image size for all gpus
        comm.synchronize()
        if training and self.iter == self.max_iter - 49990:
            meg = torch.BoolTensor(1).to(self.device)
            comm.synchronize()
            if comm.is_main_process():
                logger.info(
                    "[master] enable l1 loss now at iter: %d", self.iter
                )
                # enable l1 loss at last 50000 iterations
                meg.fill_(True)

            if comm.get_world_size() > 1:
                comm.synchronize()
                dist.broadcast(meg, 0)
            # self.head.use_l1 = meg.item()
            self.use_l1 = meg.item()
            comm.synchronize()

        labels = None
        if training:
            gt_instances = None
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(self.device) for x in batched_inputs
                ]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN,
                    "'targets' in the model inputs is now renamed to 'instances'!",
                    n=10,
                )
                gt_instances = [
                    x["targets"].to(self.device) for x in batched_inputs
                ]

            targets = [
                torch.cat(
                    [
                        instance.gt_classes.float().unsqueeze(-1),
                        instance.gt_boxes.tensor,
                    ],
                    dim=-1,
                )
                for instance in gt_instances
            ]
            labels = torch.zeros((batch_size, self.max_boxes_num, 5))
            for i, target in enumerate(targets):
                tgt = target[: self.max_boxes_num]
                labels[i][: len(tgt)] = tgt

        self.iter += 1
        return images, labels, images.image_sizes

    def forward(self, batched_inputs, compute_loss: bool = False):
        """Forward pass."""
        images, labels, image_ori_sizes = self.preprocess_image(
            batched_inputs, self.training or compute_loss
        )
        img_size = images.tensor.shape[-2:]

        #  backbone
        out_features = self.backbone(images.tensor)
        outputs = self.neck(out_features)

        outs = []
        for i, output in enumerate(outputs):
            outs.append(self.convs_list[i](output))

        predictions = [
            loss_evaluator(out, labels, img_size)
            for out, loss_evaluator in zip(outs, self.loss_evaluators)
        ]
        if self.training or compute_loss:
            losses = [pred[1] for pred in predictions]
            if self.loss_type == "v7":
                keys = [
                    "loss_iou",
                    "loss_xy",
                    "loss_wh",
                    "loss_conf",
                    "loss_cls",
                ]
            else:
                keys = [
                    "loss_x",
                    "loss_y",
                    "loss_w",
                    "loss_h",
                    "loss_conf",
                    "loss_cls",
                ]
            losses_dict = {}
            for key in keys:
                losses_dict[key] = sum(loss[key] for loss in losses)

        if self.training:
            return losses_dict

        predictions = torch.cat([pred[0] for pred in predictions], dim=1)
        # NOTE: predictions is modified in-place here. bbox coordinates (first
        # 4 values in last dim) are converted from XYWH to XYXY format.
        detections, class_logits = postprocess(
            predictions,
            self.num_classes,
            self.conf_threshold,
            self.nms_threshold,
        )

        results = []
        for idx, out in enumerate(detections):
            if out is None:
                out = torch.zeros((0, 7))
            image_size = image_ori_sizes[idx]
            result = Instances(image_size)
            result.pred_boxes = Boxes(out[:, :4])
            result.scores = out[:, 5] * out[:, 4]
            result.pred_classes = out[:, -1]
            result.obj_logits = out[:, 4]
            result.cls_logits = class_logits[idx]
            results.append(result)

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            instances = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": instances})

        if compute_loss:
            return processed_results, predictions, losses_dict
        return processed_results


class YOLOHead(nn.Module):
    """YOLOv7 head."""

    _EPS = 1e-8

    def __init__(self, cfg, anchors, level):
        """Initialize YOLOHead."""
        super().__init__()
        self.level = level
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.all_anchors = np.array(cfg.MODEL.YOLO.ANCHORS).reshape([-1, 2])
        self.anchors = anchors
        self.ref_anchors = np.zeros((len(self.all_anchors), 4))
        self.ref_anchors[:, 2:] = self.all_anchors
        self.ref_anchors = torch.from_numpy(self.ref_anchors)
        self.anchor_ratio_thresh = cfg.MODEL.YOLO.LOSS.ANCHOR_RATIO_THRESH

        self.num_anchors = len(anchors)
        # self.num_anchors = self.ref_anchors.shape[0]
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.bbox_attrs = 5 + self.num_classes

        self.iou_threshold = cfg.MODEL.YOLO.IGNORE_THRESHOLD
        # All default lambda values are 1.0
        self.lambda_xy = cfg.MODEL.YOLO.LOSS.LAMBDA_XY
        self.lambda_wh = cfg.MODEL.YOLO.LOSS.LAMBDA_WH
        self.lambda_conf = cfg.MODEL.YOLO.LOSS.LAMBDA_CONF
        self.lambda_cls = cfg.MODEL.YOLO.LOSS.LAMBDA_CLS
        self.lambda_iou = cfg.MODEL.YOLO.LOSS.LAMBDA_IOU

        self.build_target_type = (
            cfg.MODEL.YOLO.LOSS.BUILD_TARGET_TYPE
        )  # v5 or default
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.bce_loss = nn.BCELoss(reduction="mean")
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce_cls = nn.BCEWithLogitsLoss(reduction="mean")
        if self.build_target_type == "v5":
            self.cur_get_target = self.get_target_v5
        else:
            self.cur_get_target = self.get_target

    def forward(self, inputs, targets=None, image_size=(416, 416)):
        """Forward YOLOHead."""
        batch_size, _, in_h, in_w = inputs.shape
        device = inputs.device

        # image_size is input tensor size, we need convert anchor to this rel.
        stride_h = image_size[0] / in_h
        stride_w = image_size[1] / in_w
        scaled_anchors = torch.tensor(self.anchors, device=device)

        prediction = (
            inputs.view(
                batch_size, self.num_anchors, self.bbox_attrs, in_h, in_w
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # place bbox_attr to last order

        # Get outputs
        center_x = torch.sigmoid(prediction[..., 0])  # Center x
        center_y = torch.sigmoid(prediction[..., 1])  # Center y
        box_w = prediction[..., 2]  # Width
        box_h = prediction[..., 3]  # Height
        conf = prediction[..., 4]  # Conf
        pred_cls = prediction[..., 5:]  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.linspace(
            0, in_w - 1, in_w, dtype=torch.float32, device=device
        )
        grid_x = (
            grid_x.repeat(in_h, 1)
            .repeat(batch_size * self.num_anchors, 1, 1)
            .view(center_x.shape)
        )
        grid_y = torch.linspace(
            0, in_h - 1, in_h, dtype=torch.float32, device=device
        )
        grid_y = (
            grid_y.repeat(in_w, 1)
            .t()
            .repeat(batch_size * self.num_anchors, 1, 1)
            .view(center_y.shape)
        )
        # Calculate anchor w, h
        anchor_w = scaled_anchors.index_select(
            1, torch.zeros(1, dtype=torch.long, device=device)
        )
        anchor_h = scaled_anchors.index_select(
            1, torch.ones(1, dtype=torch.long, device=device)
        )
        anchor_w = (
            anchor_w.repeat(batch_size, 1)
            .repeat(1, 1, in_h * in_w)
            .view(box_w.shape)
        )
        anchor_h = (
            anchor_h.repeat(batch_size, 1)
            .repeat(1, 1, in_h * in_w)
            .view(box_h.shape)
        )
        # Add offset and scale with anchors
        pred_boxes = prediction[..., :4].clone()

        # Prevent inf box sizes
        box_w.clamp_(-16, 16)
        box_h.clamp_(-16, 16)

        # (todo) modified to adopt YOLOv5 style offsets
        pred_boxes[..., 0] = (center_x + grid_x) * stride_w
        pred_boxes[..., 1] = (center_y + grid_y) * stride_h
        pred_boxes[..., 2] = torch.exp(box_w) * anchor_w
        pred_boxes[..., 3] = torch.exp(box_h) * anchor_h

        loss = None
        conf_sigmoid = torch.sigmoid(conf)
        pred_sigmoid = torch.sigmoid(pred_cls)
        # Results
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                conf_sigmoid.view(batch_size, -1, 1),
                pred_sigmoid.view(batch_size, -1, self.num_classes),
            ),
            dim=-1,
        )

        if targets is None:
            return output, loss

        #  build target
        (
            mask,
            obj_mask,
            tx,
            ty,
            tw,
            th,
            tgt_scale,
            tcls,
        ) = self.cur_get_target(
            targets,
            pred_boxes,
            image_size,
            in_w,
            in_h,
            stride_w,
            stride_h,
        )

        if self.loss_type == "v7":
            # mask is positive samples
            loss_obj, loss_cls = 0, 0
            obj_mask = obj_mask.to(torch.bool)
            if obj_mask.any():
                loss_obj = self.bce_obj(conf[obj_mask], mask[obj_mask])
            mask = mask.to(torch.bool)
            if mask.any():
                loss_cls = self.ce_cls(pred_cls[mask], tcls[mask])

            pboxes = torch.stack([center_x, center_y, box_w, box_h], dim=-1)
            pboxes = pboxes.view(batch_size, -1, 4)
            tboxes = torch.stack([tx, ty, tw, th], dim=-1)
            tboxes = tboxes.view(batch_size, -1, 4)

            tgt_scale = tgt_scale.view(batch_size, -1, 1)
            diff = tgt_scale * (pboxes - tboxes).abs()
            loss_xy = diff[..., :2].sum(-1).mean()
            loss_wh = diff[..., 2:].sum(-1).mean()

            lbox = 0
            if mask.any():
                # replace with iou loss
                mask_viewed = mask.view(batch_size, -1)
                tgt_scale = tgt_scale.view(batch_size, -1)
                tboxes = tboxes[mask_viewed]
                pboxes = pboxes[mask_viewed]
                tgt_scale = tgt_scale[mask_viewed]
                lbox = ciou(pboxes, tboxes, sum=False)
                lbox = (lbox * tgt_scale).mean()

            loss = {
                "loss_xy": loss_xy * self.lambda_xy,
                "loss_wh": loss_wh * self.lambda_wh,
                "loss_iou": lbox * self.lambda_iou,
                "loss_conf": loss_obj * self.lambda_conf,
                "loss_cls": loss_cls * self.lambda_cls,
            }
        else:
            # FIXME: adapt to new losses (mean instead of none reduction)
            loss_conf = (obj_mask * self.bce_obj(conf, mask)).sum() / batch_size
            loss_cls = (
                self.ce_cls(pred_cls[mask == 1], tcls[mask == 1]).sum()
                / batch_size
            )

            loss_x = (
                mask * tgt_scale * self.bce_loss(center_x * mask, tx * mask)
            ).sum() / batch_size
            loss_y = (
                mask * tgt_scale * self.bce_loss(center_y * mask, ty * mask)
            ).sum() / batch_size
            loss_w = (
                mask * tgt_scale * self.l1_loss(box_w * mask, tw * mask)
            ).sum() / batch_size
            loss_h = (
                mask * tgt_scale * self.l1_loss(box_h * mask, th * mask)
            ).sum() / batch_size

            # we are not using loss_x, loss_y here, just using a simple ciou loss
            loss = {
                "loss_x": loss_x * self.lambda_xy,
                "loss_y": loss_y * self.lambda_xy,
                "loss_w": loss_w * self.lambda_wh,
                "loss_h": loss_h * self.lambda_wh,
                "loss_conf": loss_conf * self.lambda_conf,
                "loss_cls": loss_cls * self.lambda_cls,
            }

        conf = torch.sigmoid(conf)
        pred_cls = torch.sigmoid(pred_cls)
        # Results
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            dim=-1,
        )

        return output, loss

    def get_target(
        self,
        target,
        pred_boxes,
        img_size,
        in_w,
        in_h,
        stride_w,
        stride_h,
    ):
        device = pred_boxes.device
        batch_size = target.size(0)
        mask = torch.zeros(
            (batch_size, self.num_anchors, in_h, in_w), device=device
        )
        obj_mask = torch.ones_like(mask)
        tx = mask.clone()
        ty = mask.clone()
        tw = mask.clone()
        th = mask.clone()
        tgt_scale = mask.clone()
        tcls = torch.zeros(mask.shape + (self.num_classes,), device=device)

        nlabel = (target.sum(dim=2) > 0).sum(dim=1)
        gx_all = (target[:, :, 1] + target[:, :, 3]) / 2.0  # center x
        gy_all = (target[:, :, 2] + target[:, :, 4]) / 2.0  # center y
        gw_all = target[:, :, 3] - target[:, :, 1]  # width
        gh_all = target[:, :, 4] - target[:, :, 2]  # height
        gi_all = (gx_all / stride_w).to(torch.int16)
        gj_all = (gy_all / stride_h).to(torch.int16)

        num_fg = 0
        for b in range(batch_size):
            n = int(nlabel[b])
            if n == 0:
                continue

            truth_box = torch.zeros((n, 4), device=device)
            truth_box[:, 2] = gw_all[b, :n]
            truth_box[:, 3] = gh_all[b, :n]
            truth_i = gi_all[b, :n]
            truth_j = gj_all[b, :n]

            # change match strategy, by not using IoU maxium
            anchor_ious_all = bboxes_iou(
                truth_box.cpu(),
                self.ref_anchors.type_as(truth_box.cpu()),
                xyxy=False,
            )
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            # TODO: so we know which level it belongs to, 3 might be len(anchors)
            best_n = best_n_all % 3
            best_n_mask = (best_n_all // 3) == self.level

            truth_box[:n, 0] = gx_all[b, :n]
            truth_box[:n, 1] = gy_all[b, :n]
            pred_box = pred_boxes[b]

            pred_ious = bboxes_iou(pred_box.view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.iou_threshold
            pred_best_iou = pred_best_iou.view(pred_box.shape[:3])
            obj_mask[b] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for t in range(best_n.shape[0]):
                if best_n_mask[t] == 1:
                    # belong's to current level
                    gi, gj = truth_i[t], truth_j[t]
                    gx, gy = gx_all[b, t], gy_all[b, t]
                    gw, gh = gw_all[b, t], gh_all[b, t]

                    a = best_n[t]

                    # Masks
                    mask[b, a, gj, gi] = 1  # 17, 17
                    obj_mask[b, a, gj, gi] = 1
                    num_fg += 1

                    # Coordinates
                    tx[b, a, gj, gi] = gx / stride_w - gi
                    ty[b, a, gj, gi] = gy / stride_h - gj
                    # Width and height
                    tw[b, a, gj, gi] = torch.log(
                        gw / self.anchors[a][0] + self._EPS
                    )
                    th[b, a, gj, gi] = torch.log(
                        gh / self.anchors[a][1] + self._EPS
                    )

                    tgt_scale[b, a, gj, gi] = 2.0 - gw * gh / (
                        img_size[0] * img_size[1]
                    )
                    # One-hot encoding of label
                    tcls[b, a, gj, gi, int(target[b, t, 0])] = 1

        num_fg = max(num_fg, 1)
        return mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls

    def get_target_yolov5(
        self,
        target,
        pred_boxes,
        img_size,
        in_w,
        in_h,
        stride_w,
        stride_h,
    ):
        """get_target_yolov5.

        TODO(Documentation)
        """
        device = pred_boxes.device
        batch_size = target.size(0)
        mask = torch.zeros(
            (batch_size, self.num_anchors, in_h, in_w), device=device
        )
        obj_mask = torch.ones_like(mask)
        tx = mask.clone()
        ty = mask.clone()
        tw = mask.clone()
        th = mask.clone()
        tgt_scale = mask.clone()
        tcls = torch.zeros(mask.shape + (self.num_classes,), device=device)

        nlabel = (target.sum(dim=2) > 0).sum(dim=1)
        gx_all = (target[:, :, 1] + target[:, :, 3]) / 2.0  # center x
        gy_all = (target[:, :, 2] + target[:, :, 4]) / 2.0  # center y
        gw_all = target[:, :, 3] - target[:, :, 1]  # width
        gh_all = target[:, :, 4] - target[:, :, 2]  # height
        gi_all = (gx_all / stride_w).to(torch.int16)
        gj_all = (gy_all / stride_h).to(torch.int16)

        for b in range(batch_size):
            n = int(nlabel[b])
            if n == 0:
                continue

            truth_box = torch.zeros((n, 4), device=device)
            truth_box[:, 2] = gw_all[b, :n]
            truth_box[:, 3] = gh_all[b, :n]
            truth_i = gi_all[b, :n]
            truth_j = gj_all[b, :n]

            # (todo) this strategy not work, find why
            anchor_indices_mask = get_matching_anchors(
                truth_box.cpu(),
                self.ref_anchors.type_as(truth_box.cpu()),
                xyxy=False,
                anchor_ratio_thresh=self.anchor_ratio_thresh,
            )
            # one box, might have more than one anchor in all 9 anchors
            # select mask of current level
            anchor_indices_mask = anchor_indices_mask[
                :,
                self.level * self.num_anchors : self.level * self.num_anchors
                + self.num_anchors,
            ]
            # now we get boxes anchor indices, of current level
            truth_box[:n, 0] = gx_all[b, :n]
            truth_box[:n, 1] = gy_all[b, :n]
            pred_box = pred_boxes[b]

            pred_ious = bboxes_iou(pred_box.view(-1, 4), truth_box, xyxy=False)

            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.iou_threshold
            pred_best_iou = pred_best_iou.view(pred_box.shape[:3])
            obj_mask[b] = ~pred_best_iou

            if anchor_indices_mask.shape[0] == 0:
                continue

            # best_n.shape[0] is GT nums
            for t in range(anchor_indices_mask.shape[0]):
                # we already filtered in mask
                gi, gj = truth_i[t], truth_j[t]
                gx, gy = gx_all[b, t], gy_all[b, t]
                gw, gh = gw_all[b, t], gh_all[b, t]

                anchor_mask = anchor_indices_mask[t].to(torch.int)
                a = torch.argmax(anchor_mask)
                # a can not bigger than 3

                # Masks
                mask[b, a, gj, gi] = 1  # 17, 17
                obj_mask[b, a, gj, gi] = 1

                # Coordinates
                tx[b, a, gj, gi] = gx / stride_w - gi
                ty[b, a, gj, gi] = gy / stride_h - gj
                # Width and height
                tw[b, a, gj, gi] = torch.log(gw / self.anchors[a][0] + 1e-16)
                th[b, a, gj, gi] = torch.log(gh / self.anchors[a][1] + 1e-16)

                tgt_scale[b, a, gj, gi] = 2.0 - gw * gh / (
                    img_size[0] * img_size[1]
                )
                # One-hot encoding of label
                tcls[b, a, gj, gi, int(target[b, t, 0])] = 1

        return mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls


def get_matching_anchors(gt_boxes, anchors, anchor_ratio_thresh=2.1, xyxy=True):
    """Match anchors to ground-truth boxes.

    using YOLOv5 style choose anchors by given refanchors and gt_boxes
    we select anchors by comparing ratios rather than IoU
    """
    if xyxy:
        t_wh = gt_boxes[:, None, 2:] - gt_boxes[:, None, :2]
    else:
        t_wh = gt_boxes[:, None, 2:]

    ratio = t_wh[:, None, :] / anchors[:, 2:]  # wh ratio
    ratio.squeeze_(1)
    matched = torch.max(ratio, 1.0 / ratio).max(-1)[0] < anchor_ratio_thresh
    return matched


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """Postprocess the prediction."""
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    outputs = []
    class_logits = []
    for image_pred in prediction:
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        cls_logits = image_pred[:, 5 : 5 + num_classes]

        conf_mask = (
            image_pred[:, 4] * class_conf.squeeze() >= conf_thre
        ).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float()), 1
        )
        detections = detections[conf_mask]
        cls_logits = cls_logits[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        cls_logits = cls_logits[nms_out_index]
        outputs.append(detections)
        class_logits.append(cls_logits)
    return outputs, class_logits
