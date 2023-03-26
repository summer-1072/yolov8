import torch
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

    def forward(self, pred_box, pred_dist, target_box, target_gap, target_score, score_sum, mask_pos):
        weight = torch.masked_select(target_score.sum(2), mask_pos).unsqueeze(1)

        # iou loss
        iou_loss = self.build_iou_loss(pred_box[mask_pos], target_box[mask_pos], weight, score_sum)

        # dist focal loss
        dfl_loss = self.build_dfl_loss(pred_dist[mask_pos], target_gap[mask_pos], weight, score_sum)

        return iou_loss, dfl_loss


class Loss:
    def __init__(self, alpha, beta, topk, reg_max, box_w, cls_w, dfl_w, device, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.reg_max = reg_max
        self.box_w = box_w
        self.cls_w = cls_w
        self.dfl_w = dfl_w
        self.device = device
        self.eps = eps

        self.bce = nn.BCELoss(reduction='none')
        self.boxloss = BoxLoss(reg_max)

    def preprocess(self, labels, batch_size):
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
        mask = torch.gt(out.sum(2, keepdim=True), 0).to(torch.int64)

        return label_cls, label_box, mask

    def build_metrics(self, label_box, label_cls, pred_box, pred_cls):
        B, T = label_cls.shape[:2]

        index = torch.zeros([2, B, T], dtype=torch.int64)
        index[0] = torch.arange(B).view(-1, 1).repeat(1, T)
        index[1] = label_cls.squeeze(2)
        cls = pred_cls[index[0], :, index[1]]
        iou = bbox_iou(label_box.unsqueeze(2), pred_box.unsqueeze(1), 'CIoU').squeeze(3).clamp(0)
        metric = cls.pow(self.alpha) * iou.pow(self.beta)

        return cls, iou, metric

    def build_mask_pos(self, label_box, grid, mask, iou, metric):
        B, T = label_box.shape[:2]
        A = metric.shape[2]

        # match_pos
        left_top, right_bottom = label_box.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((grid.unsqueeze(0) - left_top, right_bottom - grid.unsqueeze(0)), 2).view(
            B, T, A, 4)

        match_pos = torch.gt(deltas.amin(3), self.eps).to(torch.int64)

        # top_pos
        match_metric = metric * match_pos
        top_metric, top_pos = torch.topk(match_metric, self.topk, 2)
        if len(mask):
            top_mask = mask.repeat([1, 1, self.topk]).bool()
        else:
            top_mask = (top_metric.max(2, keepdim=True).values > self.eps).repeat([1, 1, self.topk])

        top_pos = torch.where(top_mask, top_pos, 0)
        is_in_topk = F.one_hot(top_pos, A).sum(2)
        top_pos = torch.where(is_in_topk > 1, 0, is_in_topk)

        mask_pos = match_pos * top_pos * mask

        # drop duplicate
        mask_pos_sum = mask_pos.sum(1)
        if mask_pos_sum.max() > 1:
            mask_pos_sum = (mask_pos_sum.unsqueeze(1) > 1).repeat([1, T, 1])
            iou_max = iou.argmax(1)
            iou_max = F.one_hot(iou_max, T)
            iou_max = iou_max.permute(0, 2, 1)
            mask_pos = torch.where(mask_pos_sum, iou_max, mask_pos)

        mask_pos_max = mask_pos.argmax(1)
        mask_pos_sum = mask_pos.sum(1)

        return mask_pos, mask_pos_max, mask_pos_sum

    def build_targets(self, labels, pred_box, pred_cls, grid, grid_stride):
        # process label
        label_cls, label_box, mask = self.preprocess(labels, pred_cls.shape[0])

        # compute metric
        cls, iou, metric = self.build_metrics(label_box, label_cls, pred_box * grid_stride, pred_cls)

        # compute mask pos
        mask_pos, mask_pos_max, mask_pos_sum = self.build_mask_pos(label_box, grid * grid_stride, mask, iou, metric)

        # update metric
        metric = metric * mask_pos
        iou = iou * mask_pos
        metric_max = metric.amax(axis=2, keepdim=True)
        iou_max = iou.amax(axis=2, keepdim=True)

        # make index
        B, T = label_cls.shape[:2]
        num_cls = pred_cls.shape[2]
        batch_index = torch.arange(B, dtype=torch.int64, device=self.device).unsqueeze(1)
        mask_pos_max = mask_pos_max + batch_index * T

        # compute target box
        target_box = label_box.view(-1, 4)[mask_pos_max] / grid_stride

        # compute target score
        target_cls = label_cls.squeeze(2).flatten()[mask_pos_max].to(torch.int64)
        target_cls = F.one_hot(target_cls, num_cls)
        mask_label = mask_pos_sum.unsqueeze(2).repeat(1, 1, num_cls)
        target_cls = torch.where(mask_label > 0, target_cls, 0)
        norm_metric = ((iou_max * metric) / (metric_max + self.eps)).amax(1).unsqueeze(2)
        target_score = target_cls * norm_metric

        # compute target gap
        target_gap = box2gap(target_box, grid, self.reg_max)

        return target_box, target_score, target_gap, mask_pos_sum.bool()

    def __call__(self, labels, pred_box, pred_cls, pred_dist, grid, grid_stride):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        target_box, target_score, target_gap, mask_pos = self.build_targets(labels, pred_box.detach(),
                                                                            pred_cls.detach(), grid, grid_stride)

        score_sum = max(target_score.sum(), 1)

        loss[1] = self.bce(pred_cls, target_score.sigmoid()).sum() / score_sum
        loss[0], loss[2] = self.boxloss(pred_box, pred_dist, target_box, target_gap, target_score, score_sum, mask_pos)

        loss[0] *= self.box_w
        loss[1] *= self.cls_w
        loss[2] *= self.dfl_w

        return loss.sum() * pred_cls.shape[0], loss.detach()
