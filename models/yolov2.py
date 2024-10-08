"""
Full definition of a YOLOv2 model, all of it in this single file.
Ref:
1) the official Darknet implementation:
https://github.com/AlexeyAB/darknet/blob/master/src/darknet.c
https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov2-voc.cfg
https://github.com/AlexeyAB/darknet/blob/master/cfg/yolo-voc.2.0.cfg (old)
2) the official YOLOv2 paper:
https://github.com/longcw/yolo2-pytorch
"""

from pprint import pprint
import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import random

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import box_iou, box_convert, clip_boxes_to_image, nms, batched_nms
from models.darknet19 import Darknet19Config, Darknet19Backbone, Darknet19Conv2d


@dataclass
class Yolov2Config:
    img_h: int = 416
    img_w: int = 416
    n_class: int = 20
    multiscale_min_sizes: Tuple[int, ...] = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # square edge
    anchors: Tuple[Tuple[int, int], ...] = ((1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52))  # w,h relative to a cell
    n_box_per_cell: int = 5
    lambda_noobj: float = 0.5
    lambda_obj: float = 1.0
    lambda_class: float = 1.0
    lambda_coord: float = 5.0
    lambda_burnin: float = 0.01
    prob_thresh: float = 0.001
    nms_iou_thresh: float = 0.5
    # FUTURE: loss-related options not mentioned in the paper but in the AlexeyAB's darknet
    #         ref: https://github.com/AlexeyAB/darknet/issues/279#issuecomment-347002399
    anchors_burnin_n_seen_img: int = 0.01  # the number of seen imgs to burn anchors cx,cy,w,h into target
    match_by_anchors: bool = True  # whether to take anchors as predicted w,h when matching predicts to targets
                                   # (i.e., bias_match in darknet cfg)
    # match_iou_type: str = 'default'  # 'default' or 'distance'
    noobj_iou_thresh: float = 0.6  # if best iou of a predicted box with any target is less than this, it's a noobj,
                                   # (i.e., thresh in darknet cfg)
    rescore: bool = True  # whether to take the predicted iou as the target for the confidence score instead of 1.0
    # softmax_class: bool = False  # whether to use softmax for class prediction instead of mean squared error


class Yolov2Head(nn.Module):
    """
    Prediction head of YOLOv2
    """
    def __init__(self, config: Yolov2Config) -> None:
        super().__init__()
        self.config = config
        self.conv1 = Darknet19Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = Darknet19Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv3 = Darknet19Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.unfold = nn.Unfold(kernel_size=2, stride=2, padding=0)
        self.conv4 = Darknet19Conv2d(1024 + 4 * 64, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(1024, config.n_box_per_cell * (5 + config.n_class),  # out_channels is 125 by default
                               kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x: Tensor, feat: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (N, 1024, img_h / 224 * 7, img_w / 224 * 7)
            feat (Tensor): (N, 512, img_h / 224 * 14, img_w / 224 * 14)
        Returns:
            logits (Tensor): (N, n_grid_h, n_grid_w, n_box_per_cell, 5 + n_class)
        """
        # N x 1024 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)
        x = self.conv1(x)
        # N x 1024 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)
        x = self.conv2(x)
        # N x 1024 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)

        # N x 512 x 26 (or 20~38 for img_h 320~608) x 26 (or 20~38 for img_h 320~608)
        feat = self.conv3(feat)  # add a conv as darknet cfg
        # N x 64 x 26 (or 20~38 for img_h 320~608) x 26 (or 20~38 for img_h 320~608)
        feat = self.unfold(feat).reshape(feat.shape[0], 256, *x.shape[-2:])
        # N x 256 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)

        x = torch.cat([x, feat], dim=1)
        # N x 1280 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)
        x = self.conv4(x)
        # N x 1024 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)
        logits = self.conv5(x)
        # N x 125 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)
        logits = logits.permute(0, 2, 3, 1)
        # N x n_grid_h x n_grid_w x n_box_per_cell * (5 + n_class)
        logits = logits.reshape(*logits.shape[:3], self.config.n_box_per_cell, 5 + self.config.n_class)
        # N x n_grid_h x n_grid_w x n_box_per_cell x (5 + n_class)

        return logits


class Yolov2(nn.Module):
    def __init__(self, config: Yolov2Config, n_seen_img: int) -> None:
        super().__init__()
        self.config = config
        self.n_seen_img = n_seen_img
        self.backbone = Darknet19Backbone(Darknet19Config())
        self.head = Yolov2Head(config)

        # Init all weights
        self.apply(self._init_weights)

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # Disabling this init as ultralytics does gives faster startup convergence
            # torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            # if module.bias is not None:
            #     torch.nn.init.zeros_(module.bias)
            pass  # pytorch default uses kaiming_uniform_ for weight and zeros_ for bias
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # TODO: zero_init_last / trunc_normal_ / head_init_scale in timm?


    def _batched_box_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """
        Return intersection-over-union (Jaccard index) between a batch of two sets of boxes.
        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        Args:
            boxes1 (Tensor[..., N, 4]): batch of first set of boxes
            boxes2 (Tensor[..., M, 4]): batch of second set of boxes
        Returns:
            Tensor[..., N, M]: each NxM matrix containing the pairwise IoU values for every element in boxes1 & boxes2 pair
        """
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        lt = torch.max(boxes1[..., None, :2], boxes2[..., None, :, :2])
        rb = torch.min(boxes1[..., None, 2:], boxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[..., None] + area2[..., None, :] - inter
        iou = inter / union
        return iou


    def _batched_distance_box_iou(self, boxes1: Tensor, boxes2: Tensor, eps: float = 1e-16) -> Tensor:
        """
        Return distance intersection-over-union (Jaccard index) between a batch of two sets of boxes.
        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        Args:
            boxes1 (Tensor[..., N, 4]): batch of first set of boxes
            boxes2 (Tensor[..., M, 4]): batch of second set of boxes
        Returns:
            Tensor[..., N, M]: each NxM matrix containing the pairwise IoU values for every element in boxes1 & boxes2 pair
        """
        iou = self._batched_box_iou(boxes1, boxes2)
        lti = torch.min(boxes1[..., None, :2], boxes2[..., None, :, :2])
        rbi = torch.max(boxes1[..., None, 2:], boxes2[..., None, :, 2:])
        whi = (rbi - lti).clamp(min=0)
        diagonal_distance_squared = (whi[..., 0] ** 2) + (whi[..., 1] ** 2) + eps
        # Centers of boxes
        cx_1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
        cy_1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
        cx_2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
        cy_2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
        # Distance between boxes' centers squared
        centers_distance_squared = ((cx_1[..., None] - cx_2[..., None, :]) ** 2) + ((cy_1[..., None] - cy_2[..., None, :]) ** 2)
        return iou - (centers_distance_squared / diagonal_distance_squared), iou


    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Debug test case:
        # logits[:, :, :, :, :] = 0
        """
        Compute the cross entropy loss.
        Args:
            logits (Tensor): see forward
            targets (Tensor): see forward
        Returns:
            loss, loss_noobj, loss_obj, loss_class, loss_xy, loss_wh (Tensor): size(,), see forward
        """
        batch_size, n_grid_h, n_grid_w = logits.shape[:3]
        dtype, device = logits.dtype, logits.device
        anchors = torch.tensor(self.config.anchors, dtype=dtype, device=device)  # size(n_box_per_cell, 2)
        loss = torch.tensor(0.0, dtype=dtype, device=device)
        loss_noobj = torch.tensor(0.0, dtype=dtype, device=device)
        loss_obj = torch.tensor(0.0, dtype=dtype, device=device)
        loss_class = torch.tensor(0.0, dtype=dtype, device=device)
        loss_burnin = torch.tensor(0.0, dtype=dtype, device=device)
        loss_xy = torch.tensor(0.0, dtype=dtype, device=device)
        loss_wh = torch.tensor(0.0, dtype=dtype, device=device)

        # Iterate over images in the batch
        for logits_per_img, targets_per_img in zip(logits, targets):  # size(n_grid_h, n_grid_w, n_box_per_cell, (5 + n_class)); size(n_grid_h, n_grid_w, n_box_per_cell, 6)
            if self.training:
                self.n_seen_img += 1

            # All idxs
            idx_y, idx_x, idx_box = torch.meshgrid(
                torch.arange(n_grid_h, device=device),
                torch.arange(n_grid_w, device=device),
                torch.arange(self.config.n_box_per_cell, device=device),
                indexing='ij'
            )  # size(n_grid_h, n_grid_w, n_box_per_cell)

            # Idxs of obj targets (may contain two same y,x locations for two different boxes in the same cell)
            obj_targets_mask = targets_per_img[:, :, :, 4] == 1.0  # size(n_grid_h, n_grid_w, n_box_per_cell)
            obj_targets_idx_y = idx_y[obj_targets_mask]  # size(n_obj_box,)
            obj_targets_idx_x = idx_x[obj_targets_mask]  # size(n_obj_box,)
            obj_targets_idx_box = idx_box[obj_targets_mask]  # size(n_obj_box,)

            # All logits are noobj if no obj targets in this image
            if obj_targets_mask.sum() <= 0:
                conf_logits = logits_per_img[:, :, :, 4]  # size(n_grid_h, n_grid_w, n_box_per_cell)
                loss_noobj += F.mse_loss(conf_logits, torch.zeros_like(conf_logits), reduction='sum')
                if self.n_seen_img <= self.config.anchors_burnin_n_seen_img:
                    xy_logits = logits_per_img[:, :, :, :2]  # size(n_grid_h, n_grid_w, n_box_per_cell, 2)
                    loss_burnin += F.mse_loss(xy_logits, torch.full_like(xy_logits, 0.5), reduction='sum')  # t_x,t_y = 0.5 means center of cell
                    wh_logits = logits_per_img[:, :, :, 2:4]  # size(n_grid_h, n_grid_w, n_box_per_cell, 2)
                    loss_burnin += F.mse_loss(wh_logits, torch.zeros_like(wh_logits), reduction='sum')  # t_w,t_h = 0 means w,h of anchor
                continue

            # AlexeyAB's darknet implementation uses only wh to calculate iou and ignore cxcy difference
            # TODO: whether to clip_boxes_to_image, but then we need to calculate x1y1x2y2 relative to
            #       the img top-left corner, i.e., consider c_x, c_y in the paper
            # TODO: improve by referencing https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py

            # Coord of centered logits within the cells containing obj targets
            obj_logits = logits_per_img[obj_targets_idx_y, obj_targets_idx_x].detach()  # size(n_obj_box, n_box_per_cell, 5 + n_class)  # detach to avoid backprop through matching
            centered_obj_x1y1x2y2_logits = torch.stack([  # relative to the top-left corner of the cell & normalized by the grid cell w,h
                0.5 - (anchors[:, 0] * (1.0 if self.config.match_by_anchors else torch.exp(obj_logits[:, :, 2]))) / 2,
                0.5 - (anchors[:, 1] * (1.0 if self.config.match_by_anchors else torch.exp(obj_logits[:, :, 3]))) / 2,
                0.5 + (anchors[:, 0] * (1.0 if self.config.match_by_anchors else torch.exp(obj_logits[:, :, 2]))) / 2,
                0.5 + (anchors[:, 1] * (1.0 if self.config.match_by_anchors else torch.exp(obj_logits[:, :, 3]))) / 2,
            ], dim=-1)  # size(n_obj_box, n_box_per_cell, 4)

            # Coord of centered obj targets
            obj_targets = targets_per_img[obj_targets_idx_y, obj_targets_idx_x, obj_targets_idx_box]  # size(n_obj_box, 6)
            centered_obj_x1y1x2y2_targets = torch.stack([  # relative to the top-left corner of the cell & normalized by the grid cell w,h
                0.5 - obj_targets[:, 2] / 2,
                0.5 - obj_targets[:, 3] / 2,
                0.5 + obj_targets[:, 2] / 2,
                0.5 + obj_targets[:, 3] / 2,
            ], dim=-1).unsqueeze(-2)  # size(n_obj_box, 1, 4)

            # Debug msg
            # print(centered_obj_x1y1x2y2_logits[0])
            # print(centered_obj_x1y1x2y2_targets)

            # Match logits to targets within each cell containing obj targets
            obj_iou_matrix = self._batched_box_iou(centered_obj_x1y1x2y2_logits, centered_obj_x1y1x2y2_targets)  # size(n_obj_box, n_box_per_cell, 1)
            obj_iou_matrix = obj_iou_matrix.squeeze(-1)  # size(n_obj_box, n_box_per_cell)
            obj_max_iou, obj_max_logits_idx_box = obj_iou_matrix.max(dim=-1)  # size(n_obj_box,)
            # If a box logit is assigned to multiple targets, the one with the highest IoU is selected
            sorted_iou, sorted_idx = obj_max_iou.sort(dim=0, descending=True)  # size(n_obj_box,)
            sorted_targets_idx_y = obj_targets_idx_y[sorted_idx]  # size(n_obj_box,)
            sorted_targets_idx_x = obj_targets_idx_x[sorted_idx]  # size(n_obj_box,)
            sorted_targets_idx_box = obj_targets_idx_box[sorted_idx]  # size(n_obj_box,)
            sorted_logits_idx_box = obj_max_logits_idx_box[sorted_idx]  # size(n_obj_box,)
            _, matched_idx = np.unique(
                torch.stack((sorted_targets_idx_y, sorted_targets_idx_x, sorted_logits_idx_box), dim=-1).cpu().numpy(),
                return_index=True, axis=0
            )  # size(n_matched_box,)
            matched_logits_idx_box = sorted_logits_idx_box[matched_idx]  # size(n_matched_box,)
            matched_logits_idx_y = matched_targets_idx_y = sorted_targets_idx_y[matched_idx]  # size(n_matched_box,)
            matched_logits_idx_x = matched_targets_idx_x = sorted_targets_idx_x[matched_idx]  # size(n_matched_box,)
            matched_targets_idx_box = sorted_targets_idx_box[matched_idx]  # size(n_matched_box,)

            # Calculate IoU btw each pair of matched logit and target for rescore
            if self.config.rescore:
                # Coord of matched logits
                matched_logits = logits_per_img[
                    matched_logits_idx_y, matched_logits_idx_x, matched_logits_idx_box
                ].detach()  # size(n_matched_box, 5 + n_class)  # detach to avoid backprop through matching
                matched_x1y1x2y2_logits = torch.stack([  # relative to the top-left corner of the img & normalized by the grid cell w,h
                    matched_logits[:, 0] - (anchors[matched_logits_idx_box, 0] * torch.exp(matched_logits[:, 2])) / 2,
                    matched_logits[:, 1] - (anchors[matched_logits_idx_box, 1] * torch.exp(matched_logits[:, 3])) / 2,
                    matched_logits[:, 0] + (anchors[matched_logits_idx_box, 0] * torch.exp(matched_logits[:, 2])) / 2,
                    matched_logits[:, 1] + (anchors[matched_logits_idx_box, 1] * torch.exp(matched_logits[:, 3])) / 2,
                ], dim=-1).unsqueeze(-2)  # size(n_matched_box, 1, 4)

                # Coord of matched targets
                matched_targets = targets_per_img[
                    matched_targets_idx_y, matched_targets_idx_x, matched_targets_idx_box
                ]  # size(n_matched_box, 6)
                matched_x1y1x2y2_targets = torch.stack([  # relative to the top-left corner of the img & normalized by the grid cell w,h
                    matched_targets[:, 0] - matched_targets[:, 2] / 2,
                    matched_targets[:, 1] - matched_targets[:, 3] / 2,
                    matched_targets[:, 0] + matched_targets[:, 2] / 2,
                    matched_targets[:, 1] + matched_targets[:, 3] / 2,
                ], dim=-1).unsqueeze(-2)  # size(n_matched_box, 1, 4)

                # Compute IoU
                matched_iou_matrix = self._batched_box_iou(matched_x1y1x2y2_logits, matched_x1y1x2y2_targets)  # size(n_matched_box, 1, 1)
                matched_iou_matrix = matched_iou_matrix.squeeze(-1).squeeze(-1)  # size(n_matched_box,)

            # Idxs of unmatched logits
            unmatched_logits_mask = torch.ones_like(logits_per_img[:, :, :, 4]).to(torch.bool)  # size(n_grid_h, n_grid_w, n_box_per_cell)
            unmatched_logits_mask[matched_logits_idx_y, matched_logits_idx_x, matched_logits_idx_box] = False
            unmatched_logits_idx_y = idx_y[unmatched_logits_mask]  # size(n_unmatched_box,)
            unmatched_logits_idx_x = idx_x[unmatched_logits_mask]  # size(n_unmatched_box,)
            unmatched_logits_idx_box = idx_box[unmatched_logits_mask]  # size(n_unmatched_box,)

            # Coord of unmatched logits
            unmatched_logits = logits_per_img[
                unmatched_logits_idx_y, unmatched_logits_idx_x, unmatched_logits_idx_box
            ].detach()  # size(n_unmatched_box, 5 + n_class)  # detach to avoid backprop through matching
            unmatched_x1y1x2y2_logits = torch.stack([  # relative to the top-left corner of the img & normalized by the grid cell w,h
                unmatched_logits_idx_x + unmatched_logits[:, 0] - (anchors[unmatched_logits_idx_box, 0] * torch.exp(unmatched_logits[:, 2])) / 2,
                unmatched_logits_idx_y + unmatched_logits[:, 1] - (anchors[unmatched_logits_idx_box, 1] * torch.exp(unmatched_logits[:, 3])) / 2,
                unmatched_logits_idx_x + unmatched_logits[:, 0] + (anchors[unmatched_logits_idx_box, 0] * torch.exp(unmatched_logits[:, 2])) / 2,
                unmatched_logits_idx_y + unmatched_logits[:, 1] + (anchors[unmatched_logits_idx_box, 1] * torch.exp(unmatched_logits[:, 3])) / 2,
            ], dim=-1)  # size(n_unmatched_box, 4)

            # Coord of obj targets
            obj_x1y1x2y2_targets = torch.stack([  # relative to the top-left corner of the img & normalized by the grid cell w,h
                obj_targets_idx_x + obj_targets[:, 0] - obj_targets[:, 2] / 2,
                obj_targets_idx_y + obj_targets[:, 1] - obj_targets[:, 3] / 2,
                obj_targets_idx_x + obj_targets[:, 0] + obj_targets[:, 2] / 2,
                obj_targets_idx_y + obj_targets[:, 1] + obj_targets[:, 3] / 2,
            ], dim=-1)  # size(n_obj_box, 4)

            # IoU btw all unmatched logits and all obj targets
            unmatched_iou_matrix = self._batched_box_iou(unmatched_x1y1x2y2_logits, obj_x1y1x2y2_targets)  # size(n_unmatched_box, n_obj_box)

            # Select noobj logits from unmatched logits
            noobj_unmatched_idx = (unmatched_iou_matrix < self.config.noobj_iou_thresh).all(dim=-1)  # size(n_unmatched_box,)
            noobj_logits_idx_y = unmatched_logits_idx_y[noobj_unmatched_idx]  # size(n_noobj_box,)
            noobj_logits_idx_x = unmatched_logits_idx_x[noobj_unmatched_idx]  # size(n_noobj_box,)
            noobj_logits_idx_box = unmatched_logits_idx_box[noobj_unmatched_idx]  # size(n_noobj_box,)

            # Compute losses

            # Burnin anchors cx,cy,w,h to unmatched logits
            if self.n_seen_img <= self.config.anchors_burnin_n_seen_img:
                unmatched_xy_logits = logits_per_img[
                    unmatched_logits_idx_y, unmatched_logits_idx_x, unmatched_logits_idx_box, :2
                ]  # size(n_unmatched_box, 2)
                loss_burnin += F.mse_loss(unmatched_xy_logits, torch.full_like(unmatched_xy_logits, 0.5), reduction='sum')  # t_x,t_y = 0.5 means center of cell
                unmatched_wh_logits = logits_per_img[
                    unmatched_logits_idx_y, unmatched_logits_idx_x, unmatched_logits_idx_box, 2:4
                ]  # size(n_unmatched_box, 2)
                loss_burnin += F.mse_loss(unmatched_wh_logits, torch.zeros_like(unmatched_wh_logits), reduction='sum')  # t_w,t_h = 0 means w,h of anchor

            # Compute matched-box confidence loss
            matched_conf_logits = logits_per_img[matched_logits_idx_y, matched_logits_idx_x, matched_logits_idx_box, 4]  # size(n_matched_box,)
            if self.config.rescore:
                loss_obj += F.mse_loss(matched_conf_logits, matched_iou_matrix, reduction='sum')
            else:
                loss_obj += F.mse_loss(matched_conf_logits, torch.ones_like(matched_conf_logits), reduction='sum')

            # Compute matched-box class loss
            matched_class_logits = logits_per_img[matched_logits_idx_y, matched_logits_idx_x, matched_logits_idx_box, 5:]  # size(n_matched_box, n_class)
            matched_class_targets = targets_per_img[matched_targets_idx_y, matched_targets_idx_x, matched_targets_idx_box, 5].to(torch.int64)  # size(n_matched_box,)
            loss_class += F.cross_entropy(matched_class_logits, matched_class_targets, reduction='sum')

            # Compute matched-box x,y loss
            matched_xy_logits = logits_per_img[matched_logits_idx_y, matched_logits_idx_x, matched_logits_idx_box, :2]  # size(n_matched_box, 2)
            matched_xy_targets = targets_per_img[matched_targets_idx_y, matched_targets_idx_x, matched_targets_idx_box, :2]  # size(n_matched_box, 2)
            loss_xy += F.mse_loss(matched_xy_logits, matched_xy_targets, reduction='sum')

            # Compute matched-box w,h loss
            matched_wh_logits = logits_per_img[matched_logits_idx_y, matched_logits_idx_x, matched_logits_idx_box, 2:4]  # size(n_matched_box, 2)
            matched_anchors = anchors[matched_logits_idx_box]  # size(n_matched_box, 2)
            matched_wh_targets = torch.log(targets_per_img[matched_targets_idx_y, matched_targets_idx_x, matched_targets_idx_box, 2:4] / matched_anchors)  # size(n_matched_box, 2)
            loss_wh += F.mse_loss(matched_wh_logits, matched_wh_targets, reduction='sum')

            # Compute noobj-box (unmatched boxes whose max iou with all targets in the img is less than thresh) confidence loss
            noobj_conf_logits = logits_per_img[noobj_logits_idx_y, noobj_logits_idx_x, noobj_logits_idx_box, 4]  # size(n_noobj_box,)
            loss_noobj += F.mse_loss(noobj_conf_logits, torch.zeros_like(noobj_conf_logits), reduction='sum')

            # Debug msg
            # print(f"\nNum of matched-box labels: {matched_logits_idx_box.numel()}, Num of matched-cell labels: {len(torch.unique(torch.stack((matched_logits_idx_y, matched_logits_idx_x), dim=-1), dim=0))}")
            # print(f"Num of matched-box: {matched_logits_idx_box.numel()}, Num of noobj-box: {noobj_logits_idx_box.numel()}, Num of ignored box: {unmatched_logits_idx_box.numel() - noobj_logits_idx_box.numel()}\n")

        loss_burnin /= batch_size
        loss_noobj /= batch_size
        loss_obj /= batch_size
        loss_class /= batch_size
        loss_xy /= batch_size
        loss_wh /= batch_size
        loss += (self.config.lambda_burnin * loss_burnin +
                 self.config.lambda_noobj * loss_noobj + self.config.lambda_obj * loss_obj +
                 self.config.lambda_class * loss_class + self.config.lambda_coord * (loss_xy + loss_wh))
        return loss, loss_burnin, loss_noobj, loss_obj, loss_class, loss_xy, loss_wh


    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): size(N, n_grid_h, n_grid_w, n_box_per_cell, 6)
                targets[i, j, k, l, 0:4] is is the box coordinates for the l-th box in the j,k-th grid cell, 0.0~1.0
                    targets[i, j, k, l, 0:2] is the cx,cy relative to top-left corner of the j,k-th grid cell
                                             and normalized by the grid cell width,height,
                                             i.e., ground truth of sigmoid(t_x), sigmoid(t_y) in the paper
                    targets[i, j, k, l, 2:4] is the w,h normalized by the grid cell width,height,
                                             i.e., ground truth of p_w * exp(t_w), p_h * exp(t_h) in the paper
                targets[i, j, k, l, 4] is the is_obj for the l-th box in the j,k-th grid cell, 1.0 if there is an object, 0.0 otherwise
                targets[i, j, k, l, 5] is the class index for the l-th box in the j,k-th grid cell, 0.0~float(n_class-1)
        Returns:
            logits (Tensor): size(N, n_grid_h, n_grid_w, n_box_per_cell, (5 + n_class))  # FUTURE: handle n_grid_h != n_grid_w
                logits[i, j, k, l, 0:4] is the box coordinates for the l-th box in the j,k-th grid cell,
                                        i.e., sigmoid(t_x), sigmoid(t_y), t_w, t_h in the paper
                logits[i, j, k, l, 4] is the objectness confidence score for the l-th box in the j,k-th grid cell,
                                      i.e., sigmoid(t_o) in the paper
                logits[i, j, k, l, 5:5+n_class] is the class logits (before softmax) for the j,k-th grid cell
            loss (Tensor): size(,), weighted sum of the following losses
            loss_noobj (Tensor): size(,), MSE of simoid(t_o) for the boxes with no object
            loss_obj (Tensor): size(,), MSE of simoid(t_o) for the boxes with object
            loss_class (Tensor): size(,), cross entropy loss btw class logits and targets
            loss_xy (Tensor): size(,), MSE of sigmoid(t_x), sigmoid(t_y)
            loss_wh (Tensor): size(,), MSE of t_w, t_h  # TODO: how about MSE of exp(t_w), exp(t_h)?
        """
        device = imgs.device

        # Forward the Yolov2 model itself
        # N x 3 x img_h x img_w
        x, feat = self.backbone(imgs)
        # x: N x 1024 x 13 (or 10~19 for img_h 320~608) x 13 (or 10~19 for img_h 320~608)
        # feat: N x 512 x 26 (or 20~38 for img_h 320~608) x 26 (or 20~38 for img_h 320~608)
        logits = self.head(x, feat)
        # N x n_grid_h x n_grid_w x n_box_per_cell x (5 + n_class)

        logits_xy = F.sigmoid(logits[..., 0:2])
        logits_wh = torch.clone(logits[..., 2:4])
        logits_conf = F.sigmoid(logits[..., 4:5])
        logits_class = torch.clone(logits[..., 5:])
        logits = torch.cat([logits_xy, logits_wh, logits_conf, logits_class], dim=-1)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            loss, loss_burnin, loss_noobj, loss_obj, loss_class, loss_xy, loss_wh = self._compute_loss(logits, targets)
        else:
            loss, loss_burnin, loss_noobj, loss_obj, loss_class, loss_xy, loss_wh = None, None, None, None, None, None, None

        return logits, loss, loss_burnin, loss_noobj, loss_obj, loss_class, loss_xy, loss_wh


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("FUTURE: init from pretrained model")


    def configure_optimizers(self, optimizer_type, learning_rate, betas, weight_decay, device_type, use_fused):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls decay, all biases and norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        if optimizer_type == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        elif optimizer_type == 'adam':
            # Create Adam optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused Adam: {use_fused}")
        elif optimizer_type == 'sgd':
            # Create SGD optimizer
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=betas[0])
            print(f"using SGD")
        else:
            raise ValueError(f"unrecognized optimizer_type: {optimizer_type}")

        return optimizer


    def estimate_tops(self):
        """
        Estimate the number of TOPS and parameters in the model.
        """
        raise NotImplementedError("FUTURE: estimate TOPS for Yolov2 model")


    @torch.inference_mode()
    def generate(self, imgs, top_k=None):
        """
        Predict on test imgs and return the top_k predictions.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        raise NotImplementedError("FUTURE: generate for Yolov2 model")
        self.train()


    @torch.inference_mode()
    def postprocess_for_eval(self, logits, Y_supp):
        """
        Postprocess the logits and the targets for metrics calculation.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        dtype, device = logits.dtype, logits.device
        preds_for_eval, targets_for_eval = [], []
        n_grid_h, n_grid_w = logits.shape[1:3]
        img_h, img_w = round(n_grid_h / 13 * 416), round(n_grid_w / 13 * 416)
        for logits_per_img, y_supp in zip(logits, Y_supp):
            grid_y, grid_x, idx_box, idx_class= torch.meshgrid(torch.arange(n_grid_h, device=device),
                                                               torch.arange(n_grid_w, device=device),
                                                               torch.arange(self.config.n_box_per_cell, device=device),
                                                               torch.arange(self.config.n_class, device=device),
                                                               indexing='ij')
            logits_class = logits_per_img[:, :, :, 5:]
            prob_class = F.softmax(logits_class, dim=-1)

            sigmoid_t_o = logits_per_img[grid_y, grid_x, idx_box, 4]
            conf = sigmoid_t_o

            prob = prob_class * conf
            mask = (prob > self.config.prob_thresh) & (conf > 0.0)
            grid_y = grid_y[mask]
            grid_x = grid_x[mask]
            idx_box = idx_box[mask]
            idx_class = idx_class[mask]
            prob = prob[mask]

            anchors = torch.tensor(self.config.anchors, dtype=dtype, device=device)
            anchors_w = anchors[idx_box, 0]
            anchors_h = anchors[idx_box, 1]

            sigmoid_t_x = logits_per_img[grid_y, grid_x, idx_box, 0]
            sigmoid_t_y = logits_per_img[grid_y, grid_x, idx_box, 1]
            t_w = logits_per_img[grid_y, grid_x, idx_box, 2]
            t_h = logits_per_img[grid_y, grid_x, idx_box, 3]

            cx = (sigmoid_t_x + grid_x) / n_grid_w * img_w
            cy = (sigmoid_t_y + grid_y) / n_grid_h * img_h
            w = torch.exp(t_w) * anchors_w / n_grid_w * img_w
            h = torch.exp(t_h) * anchors_h / n_grid_h * img_h
            coord = torch.stack((cx, cy, w, h), dim=-1)

            boxes = clip_boxes_to_image(box_convert(coord, in_fmt='cxcywh', out_fmt='xyxy'),
                                        size=(img_h, img_w))
            scores = prob
            classes = idx_class

            boxes, scores = boxes.to(torch.float32), scores.to(torch.float32)
            keep = batched_nms(boxes, scores, classes, self.config.nms_iou_thresh)  # don't work for BFloat16

            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            preds_for_eval.append(dict(boxes=boxes, scores=scores, labels=classes))
            targets_for_eval.append(dict(boxes=y_supp['boxes'].to(device), labels=y_supp['labels'].to(device)))
        return preds_for_eval, targets_for_eval



if __name__ == '__main__':
    # Test the model by `python -m models.yolov2` from the workspace directory
    config = Yolov2Config()
    model = Yolov2(config, n_seen_img=0)
    print(model)
    print(f"num params: {model.get_num_params():,}")

    imgs = torch.randn(2, 3, config.img_h, config.img_w)

    targets = []
    n_grid_h, n_grid_w = config.img_h * 13 // 416, config.img_w * 13 // 416
    for _ in range(2 * n_grid_w * n_grid_h * config.n_box_per_cell):
        targets.extend([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, n_grid_w), random.uniform(0, n_grid_h),
                        0, random.randint(0, config.n_class - 1)])
    targets = torch.tensor(targets, dtype=torch.float32).view(2, n_grid_h, n_grid_w, config.n_box_per_cell, 6)
    targets[0, 0, 0, 0, :4] = torch.tensor([0., 0., 1.08, 1.19])
    targets[0, 0, 0, 0, 4] = 1.0
    targets[0, 0, 0, 1, :4] = torch.tensor([0., 0., 1.08, 1.19])
    targets[0, 0, 0, 1, 4] = 1.0
    targets[0, 0, 0, 2, :4] = torch.tensor([0., 0., 1.08, 1.19])
    targets[0, 0, 0, 2, 4] = 1.0
    targets[0, 0, 0, 3, :4] = torch.tensor([0., 0., 9.42, 5.11])
    targets[0, 0, 0, 3, 4] = 1.0
    targets[0, 0, 0, 4, :4] = torch.tensor([0., 0., 16.62, 10.52])
    targets[0, 0, 0, 4, 4] = 1.0
    logits, loss, _, _, _, _, _, _ = model(imgs, targets)
    # logits, loss, _, _, _, _, _, _ = model(imgs)
    print(f"logits shape: {logits.shape}")
    if loss is not None:
        print(f"loss shape: {loss.shape}")
