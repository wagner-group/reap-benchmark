"""YOLOv6 head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from yolov7.modeling.head.yolov6_head import build_effidehead_layer
from yolov7.utils.boxes import IOUlossV6, pairwise_bbox_iou


class YOLOv6Head(nn.Module):
    """YOLOv6 head."""

    def __init__(
        self,
        num_classes,
        num_anchors=1,
        num_layers=3,
        in_channels=(256, 512, 1024),
        strides=(8, 16, 32),
    ):
        """Initialize YOLOv6 head."""
        super().__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        self.channels_list = in_channels
        head_layers = build_effidehead_layer(
            in_channels, self.num_anchors, num_classes
        )
        self.det_head = Detect(
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_layers=num_layers,
            head_layers=head_layers,
        )

        self.use_l1 = False
        self.compute_loss = ComputeLoss(
            iou_type="ciou",
            strides=strides,
            num_anchors=num_anchors,
            in_channels=in_channels,
        )
        self.onnx_export = False

    def initialize_biases(self, prior_prob):
        """Initialize conv bias according to given prior prob."""
        for conv in self.cls_preds:
            bias = conv.bias.view(self.n_anchors, -1)
            bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            bias = conv.bias.view(self.n_anchors, -1)
            bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, compute_loss: bool = False):
        """Forward pass."""
        outputs = self.det_head(xin, compute_loss=compute_loss)
        if self.training or compute_loss:
            losses = self.compute_loss(outputs, labels)
        if self.training:
            return None, losses

        # pylint: disable=invalid-name,attribute-defined-outside-init
        self.hw = [x.shape[2:4] for x in outputs]
        # [batch, n_anchors_all, 85]
        # outputs = torch.cat(
        #     [x.flatten(start_dim=2) for x in outputs], dim=2
        # ).permute(0, 2, 1)
        # FIXME: needed?
        if self.decode_in_inference:
            # pylint: disable=no-member
            # output, output_origin, grid, feat_h, feat_w = self.decode_output(
            #     output, k, strides[k], dtype, device
            # )
            # return self.compute_loss.decode_output(
            #     outputs, len(xin), [8, 16, 32], dtype=outputs.dtype, device=outputs.device
            # )
            # new_outputs = []
            # for k, output in enumerate(outputs):
            #     output = self.compute_loss.decode_output(
            #         output, k, [8, 16, 32], dtype=outputs.dtype, device=outputs.device
            #     )[0]
            #     new_outputs.append(output)
            # outputs = torch.cat(new_outputs, dim=0)
            outputs = self.compute_loss.get_outputs_and_grids(
                outputs,
                [8, 16, 32],
                dtype=outputs[0].dtype,
                device=outputs[0].device,
            )[0]
        if compute_loss:
            return outputs, losses
        return outputs


class ComputeLoss(nn.Module):
    """Loss computation func.

    This func contains SimOTA and siou loss.
    """

    def __init__(
        self,
        reg_weight=5.0,
        iou_weight=3.0,
        cls_weight=1.0,
        center_radius=2.5,
        eps=1e-7,
        in_channels=[256, 512, 1024],
        strides=[8, 16, 32],
        num_anchors=1,
        iou_type="ciou",
    ):
        super().__init__()
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

        self.center_radius = center_radius
        self.eps = eps
        self.n_anchors = num_anchors
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        # Define criteria
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.iou_loss = IOUlossV6(iou_type=iou_type, reduction="mean")

    def forward(self, outputs, targets):
        dtype = outputs[0].type()
        device = targets.device
        num_classes = outputs[0].shape[-1] - 5

        (
            outputs,
            outputs_origin,
            _,
            xy_shifts,
            expanded_strides,
        ) = self.get_outputs_and_grids(outputs, self.strides, dtype, device)

        total_num_anchors = outputs.shape[1]
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        bbox_preds_org = outputs_origin[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # targets
        batch_size = bbox_preds.shape[0]
        # number of objects
        num_targets_list = (targets.sum(dim=2) > 0).sum(dim=1)

        num_fg, num_gts = 0, 0
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        for batch_idx in range(batch_size):
            num_gt = int(num_targets_list[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # EDIT: No need to scale gt_bboxes since we use absolute mode
                # gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5].mul_(
                #     gt_bboxes_scale
                # )
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]

                gt_classes = targets[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    cls_preds_per_image,
                    obj_preds_per_image,
                    expanded_strides,
                    xy_shifts,
                    num_classes,
                )
                # TODO: needed?
                # torch.cuda.empty_cache()
                num_fg += num_fg_img
                if num_fg_img > 0:
                    cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
                    obj_target = fg_mask.unsqueeze(-1)
                    reg_target = gt_bboxes_per_image[matched_gt_inds]

                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        xy_shifts=xy_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            l1_targets.append(l1_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        l1_targets = torch.cat(l1_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        # loss
        loss_iou = self.iou_loss(
            bbox_preds.view(-1, 4)[fg_masks].T, reg_targets
        )
        loss_l1 = self.l1_loss(bbox_preds_org.view(-1, 4)[fg_masks], l1_targets)

        loss_obj = self.bcewithlog_loss(
            obj_preds.view(-1, 1), obj_targets.float()
        )
        loss_cls = self.ce_loss(
            cls_preds.view(-1, num_classes)[fg_masks], cls_targets
        )
        return self.reg_weight * loss_iou, loss_l1, loss_obj, loss_cls

    def decode_output(self, output, k, stride, dtype, device):
        grid = self.grids[k].to(device)
        batch_size = output.shape[0]
        hsize, wsize = output.shape[2:4]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = (
                torch.stack((xv, yv), 2)
                .view(1, 1, hsize, wsize, 2)
                .type(dtype)
                .to(device)
            )
            self.grids[k] = grid

        output = output.reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        output_origin = output.clone()
        grid = grid.view(1, -1, 2)

        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, output_origin, grid, hsize, wsize

    def get_outputs_and_grids(self, outputs, strides, dtype, device):
        xy_shifts = []
        expanded_strides = []
        outputs_new = []
        outputs_origin = []

        for k, output in enumerate(outputs):
            output, output_origin, grid, feat_h, feat_w = self.decode_output(
                output, k, strides[k], dtype, device
            )

            xy_shift = grid
            expanded_stride = torch.full(
                (1, grid.shape[1], 1),
                strides[k],
                dtype=grid.dtype,
                device=grid.device,
            )

            xy_shifts.append(xy_shift)
            expanded_strides.append(expanded_stride)
            outputs_new.append(output)
            outputs_origin.append(output_origin)

        xy_shifts = torch.cat(xy_shifts, 1)  # [1, n_anchors_all, 2]
        expanded_strides = torch.cat(
            expanded_strides, 1
        )  # [1, n_anchors_all, 1]
        outputs_origin = torch.cat(outputs_origin, 1)
        outputs = torch.cat(outputs_new, 1)

        feat_h *= strides[-1]
        feat_w *= strides[-1]
        gt_bboxes_scale = torch.Tensor(
            [[feat_w, feat_h, feat_w, feat_h]]
        ).type_as(outputs)

        return (
            outputs,
            outputs_origin,
            gt_bboxes_scale,
            xy_shifts,
            expanded_strides,
        )

    def get_l1_target(self, l1_target, gt, stride, xy_shifts, eps=1e-8):

        l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
        l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        xy_shifts,
        num_classes,
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            xy_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # cost
        pair_wise_ious = pairwise_bbox_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, box_format="xywh"
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = cls_preds_.float().sigmoid_().unsqueeze(0).repeat(
                num_gt, 1, 1
            ) * obj_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_, obj_preds_

        cost = (
            self.cls_weight * pair_wise_cls_loss
            + self.iou_weight * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask
        )

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        xy_shifts,
        total_num_anchors,
        num_gt,
    ):
        # FIXME: why can just select [0]
        expanded_strides_per_image = expanded_strides[0]
        xy_shifts_per_image = xy_shifts[0] * expanded_strides_per_image
        xy_centers_per_image = (
            (xy_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1, 1)
        )  # [n_anchor, 2] -> [n_gt, n_anchor, 2]

        gt_bboxes_per_image_lt = (
            (gt_bboxes_per_image[:, 0:2] - 0.5 * gt_bboxes_per_image[:, 2:4])
            .unsqueeze(1)
            .repeat(1, total_num_anchors, 1)
        )
        gt_bboxes_per_image_rb = (
            (gt_bboxes_per_image[:, 0:2] + 0.5 * gt_bboxes_per_image[:, 2:4])
            .unsqueeze(1)
            .repeat(1, total_num_anchors, 1)
        )  # [n_gt, 2] -> [n_gt, n_anchor, 2]

        b_lt = xy_centers_per_image - gt_bboxes_per_image_lt
        b_rb = gt_bboxes_per_image_rb - xy_centers_per_image
        bbox_deltas = torch.cat([b_lt, b_rb], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # in fixed center
        expanded_strides_per_image.unsqueeze_(0)
        gt_bboxes_per_image_lt = (gt_bboxes_per_image[:, 0:2]).unsqueeze(1)
        gt_bboxes_per_image_lt = gt_bboxes_per_image_lt.repeat(
            1, total_num_anchors, 1
        )
        gt_bboxes_per_image_lt -= (
            self.center_radius * expanded_strides_per_image
        )
        gt_bboxes_per_image_rb = gt_bboxes_per_image_lt.clone()
        gt_bboxes_per_image_rb += (
            self.center_radius * expanded_strides_per_image
        )

        c_lt = xy_centers_per_image - gt_bboxes_per_image_lt
        c_rb = gt_bboxes_per_image_rb - xy_centers_per_image
        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor]
            & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(
        self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask
    ):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        )


class Detect(nn.Module):
    """Efficient Decoupled Head.

    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    """

    def __init__(
        self,
        num_classes=80,
        num_anchors=1,
        num_layers=3,
        inplace=True,
        head_layers: nn.Module | None = None,
    ):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(num_anchors, (list, tuple)):
            self.na = len(num_anchors[0]) // 2
        else:
            self.na = num_anchors
        self.anchors = num_anchors
        self.prior_prob = 1e-2
        self.inplace = inplace
        self.grid = [torch.zeros(1)] * num_layers
        # strides computed during build
        self.register_buffer("stride", torch.tensor([8, 16, 32]))

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 6
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
            self.obj_preds.append(head_layers[idx + 5])

    def initialize_biases(self):
        for conv in self.cls_preds:
            bias = conv.bias.view(self.na, -1)
            bias.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            bias = conv.bias.view(self.na, -1)
            bias.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def forward(self, inputs: torch.Tensor, compute_loss: bool = False):
        z = []
        for i in range(self.nl):
            inputs[i] = self.stems[i](inputs[i])
            cls_x = inputs[i]
            reg_x = inputs[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)
            if self.training or compute_loss:
                inputs[i] = torch.cat([reg_output, obj_output, cls_output], 1)
                bs, _, ny, nx = inputs[i].shape
                inputs[i] = (
                    inputs[i]
                    .view(bs, self.na, self.no, ny, nx)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                continue
            # TODO: use this during attack or above
            # y = torch.cat(
            #     [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            # )
            y = torch.cat(
                [
                    reg_output,
                    obj_output.sigmoid(),
                    cls_output.softmax(dim=1),
                ],
                1,
            )
            bs, _, ny, nx = y.shape
            y = (
                y.view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if self.grid[i].shape[2:4] != y.shape[2:4]:
                yv, xv = torch.meshgrid(
                    [
                        torch.arange(ny).to(self.stride.device),
                        torch.arange(nx).to(self.stride.device),
                    ]
                )
                self.grid[i] = (
                    torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float()
                )
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[
                    i
                ]  # xy
                y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
            else:
                xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, self.no))

        if self.training or compute_loss:
            return inputs
        return torch.cat(z, dim=1)
