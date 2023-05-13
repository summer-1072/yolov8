import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from box import bbox_iou, box2gap


class BoxLoss(nn.Module):
    def __init__(self, reg_max):
        super().__init__()
        self.reg_max = reg_max

    def build_iou_loss(self, pred_box, target_box, weight, score_sum):
        iou = bbox_iou(pred_box, target_box, 'CIoU')

        return ((1.0 - iou) * weight).sum() / score_sum

    def build_dfl_loss(self, pred_dist, target_gap, weight, score_sum):
        pred_dist = pred_dist.view(-1, self.reg_max)
        gap_min = target_gap.long()
        gap_max = gap_min + 1
        weight_min = gap_max - target_gap
        weight_max = 1 - weight_min

        loss1 = F.cross_entropy(pred_dist, gap_min.view(-1), reduction="none").view(gap_min.shape) * weight_min
        loss2 = F.cross_entropy(pred_dist, gap_max.view(-1), reduction="none").view(gap_max.shape) * weight_max

        return ((loss1 + loss2).mean(-1, keepdim=True) * weight).sum() / score_sum

    def forward(self, pred_box, pred_dist, target_box, target_gap, target_score, score_sum, mask):
        weight = torch.masked_select(target_score.sum(2), mask).unsqueeze(1)

        # iou loss
        iou_loss = self.build_iou_loss(pred_box[mask], target_box[mask], weight, score_sum)

        # dist focal loss
        dfl_loss = self.build_dfl_loss(pred_dist[mask], target_gap[mask], weight, score_sum)

        return iou_loss, dfl_loss


class LossFun:
    def __init__(self, alpha, beta, topk, box_w, cls_w, dfl_w, reg_max, device, eps=1e-9):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.box_w = box_w
        self.cls_w = cls_w
        self.dfl_w = dfl_w
        self.reg_max = reg_max
        self.device = device
        self.eps = eps

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.boxloss = BoxLoss(reg_max)

        self.names = ['box_loss', 'cls_loss', 'dfl_loss']

    def build_label(self, labels, batch_size):
        if labels.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            indices = labels[:, 0]
            index, count = indices.unique(return_counts=True)
            out = torch.zeros(batch_size, count.max(), 5, device=self.device)
            for index in range(batch_size):
                matches = indices == index
                num = matches.sum()
                if num:
                    out[index, :num] = labels[matches, 1:]

        label_cls, label_box = out.split((1, 4), 2)
        label_mask = torch.gt(out.sum(2, keepdim=True), 0).to(torch.long)

        return label_cls, label_box, label_mask

    def build_mask(self, label_cls, label_box, label_mask, pred_cls, pred_box, grid):
        B, T = label_box.shape[:2]
        A = grid.shape[0]

        # match mask
        left_top, right_bottom = label_box.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((grid.unsqueeze(0) - left_top, right_bottom - grid.unsqueeze(0)), 2).view(B, T, A, 4)

        match_mask = torch.gt(deltas.amin(3), self.eps).to(torch.long)

        # cal metric
        mask = (label_mask * match_mask).bool()
        cls = torch.zeros([B, T, A], dtype=pred_cls.dtype, device=self.device)
        iou = torch.zeros([B, T, A], dtype=pred_box.dtype, device=self.device)

        index = torch.zeros([2, B, T], dtype=torch.long)
        index[0] = torch.arange(B).view(-1, 1).repeat(1, T)
        index[1] = label_cls.squeeze(2)
        cls[mask] = pred_cls[index[0], :, index[1]][mask]

        mask_label_box = label_box.unsqueeze(2).repeat(1, 1, A, 1)[mask]
        mask_pred_box = pred_box.unsqueeze(1).repeat(1, T, 1, 1)[mask]
        iou[mask] = bbox_iou(mask_label_box, mask_pred_box, 'CIoU').squeeze(1).clamp(0)

        metric = cls.pow(self.alpha) * iou.pow(self.beta)

        # top mask
        top_metric, top_index = torch.topk(metric, self.topk, dim=2, largest=True)

        if len(label_mask):
            top_mask = label_mask.repeat([1, 1, self.topk]).bool()
        else:
            top_mask = (top_metric.max(2, keepdim=True).values > self.eps).repeat([1, 1, self.topk])

        top_index[~top_mask] = 0
        is_in_topk = torch.zeros(metric.shape, dtype=torch.long, device=self.device)
        for i in range(self.topk):
            is_in_topk += F.one_hot(top_index[:, :, i], A)

        top_mask = torch.where(is_in_topk > 1, 0, is_in_topk)

        mask = label_mask * match_mask * top_mask

        # drop duplicate
        mask_sum = mask.sum(1)
        if mask_sum.max() > 1:
            mask_sum = (mask_sum.unsqueeze(1) > 1).repeat([1, T, 1])
            iou_max = iou.argmax(1)
            iou_max = F.one_hot(iou_max, T)
            iou_max = iou_max.permute(0, 2, 1)
            mask = torch.where(mask_sum, iou_max, mask)

        return iou, metric, mask

    def build_targets(self, labels, pred_cls, pred_box, grid, grid_stride):
        if len(labels) == 0:
            return torch.zeros(pred_cls.shape).to(self.device), None, None, None

        # label cls、box、mask
        label_cls, label_box, label_mask = self.build_label(labels, pred_cls.shape[0])

        # iou, metric, mask
        iou, metric, mask = self.build_mask(label_cls, label_box, label_mask, pred_cls,
                                            pred_box * grid_stride, grid * grid_stride)

        B, T = label_cls.shape[:2]
        num_cls = pred_cls.shape[2]

        # norm metric
        iou *= mask
        metric *= mask
        iou_max = iou.amax(axis=2, keepdim=True)
        metric_max = metric.amax(axis=2, keepdim=True)
        norm_metric = (metric * iou_max / (metric_max + self.eps)).amax(1).unsqueeze(2)

        # target cls
        batch_index = torch.arange(B, dtype=torch.long, device=self.device).unsqueeze(1)
        mask_max = mask.argmax(1) + batch_index * T
        target_cls = label_cls.flatten()[mask_max].clamp(0).to(torch.long)

        # target score
        target_score = F.one_hot(target_cls, num_cls)
        mask_sum = mask.sum(1).unsqueeze(2).repeat(1, 1, num_cls)
        target_score = torch.where(mask_sum > 0, target_score, 0)
        target_score = target_score * norm_metric

        # target box
        target_box = label_box.view(-1, 4)[mask_max] / grid_stride

        # target gap
        target_gap = box2gap(target_box, grid, self.reg_max)

        return target_score, target_box, target_gap, mask.sum(1).bool()

    def __call__(self, labels, pred_cls, pred_box, pred_dist, grid, grid_stride):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        target_score, target_box, target_gap, mask = self.build_targets(labels, pred_cls.detach().sigmoid(),
                                                                        pred_box.detach().type(labels.dtype),
                                                                        grid, grid_stride)

        score_sum = max(target_score.sum(), 1)
        loss[1] = self.bce(pred_cls, target_score).sum() / score_sum

        if mask is not None:
            loss[0], loss[2] = self.boxloss(pred_box, pred_dist, target_box, target_gap, target_score, score_sum, mask)

        loss[0] *= self.box_w
        loss[1] *= self.cls_w
        loss[2] *= self.dfl_w

        return loss.sum() * pred_cls.shape[0], loss.detach().cpu().numpy()
