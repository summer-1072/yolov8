import torch
from torch import nn
from box import bbox_iou
import torch.nn.functional as F


class BoxLoss(nn.Module):
    def __init__(self, reg_max):
        super(BoxLoss).__init__()
        self.reg_max = reg_max

    def forward(self, targets, preds, gpoints, mask_pos):
        pass


class Loss:
    def __init__(self, alpha, beta, topk, device, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.device = device
        self.eps = eps

        self.bce = nn.BCELoss(reduction='none')

    def format_labels(self, labels, preds):
        B = len(preds)
        if labels.shape[0] == 0:
            out = torch.zeros(B, 0, 5, device=self.device)
        else:
            indices = labels[:, 0]
            index, count = indices.unique(return_counts=True)
            out = torch.zeros(B, count.max(), 5, device=self.device)
            for index in range(B):
                matches = indices == index
                num = matches.sum()
                if num:
                    out[index, :num] = labels[matches, 1:]

        mask = torch.gt(out.sum(2, keepdim=True), 0).to(torch.int64)

        return out, mask

    def build_metrics(self, labels, preds):
        B, T = labels.shape[:2]

        index = torch.zeros([2, B, T], dtype=torch.int64)
        index[0] = torch.arange(B).view(-1, 1).repeat(1, T)
        index[1] = labels[:, :, 0]
        cls = preds[:, :, 0:-4][index[0], :, index[1]]
        iou = bbox_iou(labels[:, :, 1:5].unsqueeze(2), preds[:, :, -4:].unsqueeze(1), 'CIoU').squeeze(3).clamp(0)
        metric = cls.pow(self.alpha) * iou.pow(self.beta)

        return cls, iou, metric

    def build_mask_pos(self, labels, gpoints, mask, iou, metric):
        B, T = labels.shape[:2]
        A = metric.shape[2]

        # match_pos
        left_top, right_bottom = labels[:, :, 1:5].view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((gpoints.unsqueeze(0) - left_top, right_bottom - gpoints.unsqueeze(0)), 2).view(
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

        max_pos_max = mask_pos.argmax(1)
        mask_pos_sum = mask_pos.sum(1)

        return mask_pos, max_pos_max, mask_pos_sum

    def build_targets(self, labels, preds, gpoints, gstrides, img_size):
        # format label
        labels, mask = self.format_labels(labels, preds)

        # scale param
        s_labels = torch.cat([labels[:, :, :-4], labels[:, :, -4:] * img_size[[1, 0, 1, 0]]], 2)
        s_preds = torch.cat([preds[:, :, 4:].sigmoid(), preds[:, :, 0:4] * gstrides], 2)
        s_gpoints = gpoints * gstrides

        # compute metric
        cls, iou, metric = self.build_metrics(s_labels, s_preds)

        # compute mask pos
        mask_pos, max_pos_max, mask_pos_sum = self.build_mask_pos(s_labels, s_gpoints, mask, iou, metric)

        # update metric
        metric = metric * mask_pos
        iou = iou * mask_pos
        metric_max = metric.amax(axis=2, keepdim=True)
        iou_max = iou.amax(axis=2, keepdim=True)

        # compute target score
        B, T = s_labels.shape[:2]
        num_cls = s_preds.shape[2] - 4
        batch_index = torch.arange(B, dtype=torch.int64, device=self.device).unsqueeze(1)
        max_pos_max = max_pos_max + batch_index * T
        target_labels = s_labels[:, :, 0].flatten()[max_pos_max].to(torch.int64)
        target_labels = F.one_hot(target_labels, num_cls)
        mask_labels = mask_pos_sum.unsqueeze(2).repeat(1, 1, num_cls)
        target_labels = torch.where(mask_labels > 0, target_labels, 0)
        norm_metric = ((iou_max * metric) / (metric_max + self.eps)).amax(1).unsqueeze(2)
        target_scores = target_labels * norm_metric

        # compute target box
        target_boxes = s_labels[:, :, 1:].view(-1, 4)[max_pos_max] / gstrides

        targets = torch.cat([target_scores, target_boxes], dim=2)

        weight = torch.masked_select(target_scores.sum(-1), mask_pos_sum.bool())
        iou = bbox_iou(preds[:, :, -4:][mask_pos_sum.bool()], targets[:, :, -4:][mask_pos_sum.bool()])

        return targets, mask_pos_sum.bool()

    def __call__(self, labels, preds, gpoints, gstrides, img_size):
        # labels: B、T、1 + 4
        # preds: B、A、num_cls + 4
        # gpoints: A、2
        # gstrides: A、1

        loss = torch.zeros(3, device=self.device)  # cls, box, dfl

        targets, mask_pos = self.build_targets(labels, preds, gpoints, gstrides, img_size)

        scores_sum = max(targets[:, :, :-4].sum(), 1)

        loss[0] = self.bce(preds[:, :, :-4].sigmoid(), targets[:, :, :-4]).sum()

        if mask_pos.sum():
            pass

        # loss[0] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE


labels = torch.tensor([
    [0, 0, 0.1, 0.1, 0.6, 0.6],
    [0, 1, 0.4, 0.4, 0.9, 0.9],
    [1, 0, 0.1, 0.4, 0.3, 0.6],
    [1, 1, 0.4, 0.1, 0.9, 0.5],
    [1, 2, 0.1, 0.7, 0.9, 0.9]])

preds = torch.tensor([
    [[0.05, 0.05, 1.05, 1.05, 0.10, 0.20, 0.30],
     [1.05, 0.05, 2.05, 1.05, 0.15, 0.25, 0.35],
     [2.05, 0.05, 3.05, 1.05, 0.20, 0.30, 0.40],
     [0.05, 1.05, 1.05, 2.05, 0.25, 0.35, 0.45],
     [1.05, 1.05, 2.05, 2.05, 0.30, 0.40, 0.50],
     [2.05, 1.05, 3.05, 2.05, 0.35, 0.45, 0.55],
     [0.05, 2.05, 1.05, 3.05, 0.40, 0.50, 0.60],
     [1.05, 2.05, 2.05, 3.05, 0.45, 0.55, 0.65],
     [2.05, 2.05, 3.05, 3.05, 0.50, 0.60, 0.70],

     [0.05, 0.05, 1.05, 1.05, 0.10, 0.20, 0.30],
     [1.05, 0.05, 2.05, 1.05, 0.15, 0.25, 0.35],
     [0.05, 1.05, 1.05, 2.05, 0.20, 0.30, 0.40],
     [1.05, 1.05, 2.05, 2.05, 0.25, 0.35, 0.45]],

    [[0.10, 0.20, 0.30, 0.05, 0.05, 1.05, 1.05],
     [0.15, 0.25, 0.35, 1.05, 0.05, 2.05, 1.05],
     [0.20, 0.30, 0.40, 2.05, 0.05, 3.05, 1.05],
     [0.25, 0.35, 0.45, 0.05, 1.05, 1.05, 2.05],
     [0.30, 0.40, 0.50, 1.05, 1.05, 2.05, 2.05],
     [0.35, 0.45, 0.55, 2.05, 1.05, 3.05, 2.05],
     [0.40, 0.50, 0.60, 0.05, 2.05, 1.05, 3.05],
     [0.45, 0.55, 0.65, 1.05, 2.05, 2.05, 3.05],
     [0.50, 0.60, 0.70, 2.05, 2.05, 3.05, 3.05],

     [0.10, 0.20, 0.30, 0.05, 0.05, 1.05, 1.05],
     [0.15, 0.25, 0.35, 1.05, 0.05, 2.05, 1.05],
     [0.20, 0.30, 0.40, 0.05, 1.05, 1.05, 2.05],
     [0.25, 0.35, 0.45, 1.05, 1.05, 2.05, 2.05]]
])

gpoints = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5],
                        [0.5, 0.5], [1.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5]
                        ])

gstrides = torch.tensor([[4], [4], [4], [4], [4], [4], [4], [4], [4], [6], [6], [6], [6]])
img_size = torch.tensor([12, 12])

loss = Loss(1, 1, 3, 'cpu')
loss(labels.detach(), preds.detach(), gpoints, gstrides, img_size)
