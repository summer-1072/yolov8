import torch
from torch import nn
import torch.nn.functional as F
from box import bbox_iou, box2gap


class BoxLoss(nn.Module):
    def __init__(self, reg_max):
        super(BoxLoss).__init__()
        self.reg_max = reg_max

    def forward(self, pred_box, pred_dist, target_box, target_gap, target_score, score_sum, mask_pos):
        # iou loss
        weight = torch.masked_select(target_score.sum(2), mask_pos).unsqueeze(1)
        pos_pred_box, pos_target_box = pred_box[mask_pos], target_box[mask_pos]
        iou = bbox_iou(pos_pred_box, pos_target_box, 'CIoU')
        iou_loss = ((1.0 - iou) * weight).sum() / score_sum

        # dist focal loss
        pos_pred_dist, pos_target_gap = pred_dist[mask_pos].view(-1, self.reg_max), target_gap[mask_pos]

        pos_target_gap.long()


class Loss:
    def __init__(self, alpha, beta, topk, reg_max, device, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.reg_max = reg_max
        self.device = device
        self.eps = eps

        self.bce = nn.BCELoss(reduction='none')

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

        max_pos_max = mask_pos.argmax(1)
        mask_pos_sum = mask_pos.sum(1)

        return mask_pos, max_pos_max, mask_pos_sum

    def build_targets(self, labels, pred_box, pred_cls, grid, grid_stride, img_size):
        # process label
        label_cls, label_box, mask = self.preprocess(labels, pred_cls.shape[0])

        # scale data
        scale_label_box = label_box * img_size[[1, 0, 1, 0]]
        scale_pred_box = pred_box * grid_stride
        scale_grid = grid * grid_stride

        # compute metric
        cls, iou, metric = self.build_metrics(scale_label_box, label_cls, scale_pred_box, pred_cls)

        # compute mask pos
        mask_pos, max_pos_max, mask_pos_sum = self.build_mask_pos(scale_label_box, scale_grid, mask, iou, metric)

        # update metric
        metric = metric * mask_pos
        iou = iou * mask_pos
        metric_max = metric.amax(axis=2, keepdim=True)
        iou_max = iou.amax(axis=2, keepdim=True)

        # make index
        B, T = label_cls.shape[:2]
        num_cls = pred_cls.shape[2]
        batch_index = torch.arange(B, dtype=torch.int64, device=self.device).unsqueeze(1)
        max_pos_max = max_pos_max + batch_index * T

        # compute target box
        target_box = scale_label_box.view(-1, 4)[max_pos_max] / grid_stride

        # compute target score
        target_cls = label_cls.squeeze(2).flatten()[max_pos_max].to(torch.int64)
        target_cls = F.one_hot(target_cls, num_cls)
        mask_label = mask_pos_sum.unsqueeze(2).repeat(1, 1, num_cls)
        target_cls = torch.where(mask_label > 0, target_cls, 0)
        norm_metric = ((iou_max * metric) / (metric_max + self.eps)).amax(1).unsqueeze(2)
        target_score = target_cls * norm_metric

        # compute target gap
        target_gap = box2gap(target_box, grid, self.reg_max)

        return target_box, target_score, target_gap, mask_pos_sum.bool()

    def __call__(self, labels, pred_box, pred_cls, pred_dist, grid, grid_stride, img_size):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        target_box, target_score, target_gap, mask_pos = self.build_targets(labels, pred_box, pred_cls,
                                                                            grid, grid_stride, img_size)

        score_sum = max(target_score.sum(), 1)

        loss[1] = self.bce(pred_cls, target_score).sum() / score_sum

        a = target_gap[mask_pos]
        # print(a)

        # tl = a.long()  # target left
        # tr = tl + 1  # target right
        # wl = tr - a  # weight left
        # wr = 1 - wl  # weight right
        #
        # print(pred_dist[mask_pos].shape)
        # print(tl.view(-1).shape)
        #
        F.cross_entropy(pred_dist[mask_pos].view(-1, self.reg_max), tl.view(-1), reduction="none")


labels = torch.tensor([
    [0, 0, 0.1, 0.1, 0.6, 0.6],
    [0, 1, 0.4, 0.4, 0.9, 0.9],
    [1, 0, 0.1, 0.4, 0.3, 0.6],
    [1, 1, 0.4, 0.1, 0.9, 0.5],
    [1, 2, 0.1, 0.7, 0.9, 0.9]])

pred_box = torch.tensor([
    [[0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [2.05, 0.05, 3.05, 1.05],
     [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05], [2.05, 1.05, 3.05, 2.05],
     [0.05, 2.05, 1.05, 3.05], [1.05, 2.05, 2.05, 3.05], [2.05, 2.05, 3.05, 3.05],
     [0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05]],

    [[0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [2.05, 0.05, 3.05, 1.05],
     [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05], [2.05, 1.05, 3.05, 2.05],
     [0.05, 2.05, 1.05, 3.05], [1.05, 2.05, 2.05, 3.05], [2.05, 2.05, 3.05, 3.05],
     [0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05]]
])

pred_cls = torch.tensor([
    [[0.10, 0.20, 0.30], [0.15, 0.25, 0.35], [0.20, 0.30, 0.40],
     [0.25, 0.35, 0.45], [0.30, 0.40, 0.50], [0.35, 0.45, 0.55],
     [0.40, 0.50, 0.60], [0.45, 0.55, 0.65], [0.50, 0.60, 0.70],
     [0.10, 0.20, 0.30], [0.15, 0.25, 0.35], [0.20, 0.30, 0.40], [0.25, 0.35, 0.45]],

    [[0.10, 0.20, 0.30], [0.15, 0.25, 0.35], [0.20, 0.30, 0.40],
     [0.25, 0.35, 0.45], [0.30, 0.40, 0.50], [0.35, 0.45, 0.55],
     [0.40, 0.50, 0.60], [0.45, 0.55, 0.65], [0.50, 0.60, 0.70],
     [0.10, 0.20, 0.30], [0.15, 0.25, 0.35], [0.20, 0.30, 0.40], [0.25, 0.35, 0.45]]
])

pred_dist = torch.tensor([
    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],

    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]]
])

grid = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                     [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                     [0.5, 2.5], [1.5, 2.5], [2.5, 2.5],
                     [0.5, 0.5], [1.5, 0.5],
                     [0.5, 1.5], [1.5, 1.5]
                     ])

grid_stride = torch.tensor([[4], [4], [4], [4], [4], [4], [4], [4], [4], [6], [6], [6], [6]])

img_size = torch.tensor([12, 12])

loss = Loss(1, 1, 3, 4, 'cpu')
loss(labels, pred_box, pred_cls, pred_dist, grid, grid_stride, img_size)
