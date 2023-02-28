import torch
import torch.nn.functional as F
from box import xywh2xyxy, bbox_iou


class Loss:
    def __init__(self, alpha, beta, topk, device, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.device = device
        self.eps = eps

    def prepare_params(self, labels, preds, gpoints, gstrides, img_size):
        # format and scale labels
        B = len(preds)
        if labels.shape[0] == 0:
            img_labels = torch.zeros(B, 0, 5, device=self.device)
        else:
            indices = labels[:, 0]
            index, count = indices.unique(return_counts=True)
            img_labels = torch.zeros(B, count.max(), 5, device=self.device)
            for index in range(B):
                matches = indices == index
                num = matches.sum()
                if num:
                    img_labels[index, :num] = labels[matches, 1:]

            img_labels[:, :, 1:5] = xywh2xyxy(img_labels[:, :, 1:5] * img_size[[1, 0, 1, 0]])

        # scale preds
        img_preds = torch.cat([preds[:, :, :-4], preds[:, :, -4:] * gstrides], 2)

        # scale grids
        img_gpoints = gpoints * gstrides

        # label mask
        mask = torch.gt(img_labels.sum(2, keepdim=True), 0).to(torch.int64)

        return img_labels, img_preds, img_gpoints, mask

    def build_metrics(self, labels, preds):
        B, T = labels.shape[:2]

        index = torch.zeros([2, B, T], dtype=torch.int64)
        index[0] = torch.arange(B).view(-1, 1).repeat(1, T)
        index[1] = labels[:, :, 0]
        cls = preds[:, :, 0:-4][index[0], :, index[1]]
        iou = bbox_iou(labels[:, :, 1:5].unsqueeze(2), preds[:, :, -4:].unsqueeze(1), 'CIoU').squeeze(3).clamp(0)
        metric = cls.pow(self.alpha) * iou.pow(self.beta)

        return cls, iou, metric

    def build_mask_pos(self, labels, cpoints, mask, iou, metric):
        B, T = labels.shape[:2]
        A = metric.shape[2]

        # match_pos
        left_top, right_bottom = labels[:, :, 1:5].view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((cpoints.unsqueeze(0) - left_top, right_bottom - cpoints.unsqueeze(0)), 2).view(
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

    def build_

    def build_targets(self, labels, preds, gpoints, gstrides, img_size):
        img_labels, img_preds, img_gpoints, mask = self.prepare_params(labels, preds, gpoints, gstrides, img_size)

        cls, iou, metric = self.build_metrics(img_labels, img_preds)

        mask_pos, max_pos_max, mask_pos_sum = self.build_mask_pos(img_labels, img_gpoints, mask, iou, metric)

        metric = metric * mask_pos
        iou = iou * mask_pos
        metric_max = metric.amax(axis=2, keepdim=True)
        iou_max = iou.amax(axis=2, keepdim=True)

        B, T = img_labels.shape[:2]
        batch_index = torch.arange(B, dtype=torch.int64, device=self.device).unsqueeze(1)
        max_pos_max = max_pos_max + batch_index * T

        target_labels = img_labels[:, :, 0].flatten()[max_pos_max].to(torch.int64)
        target_labels = F.one_hot(target_labels, self.num_cls)
        mask_labels = mask_pos_sum.unsqueeze(2).repeat(1, 1, self.num_cls)
        target_labels = torch.where(mask_labels > 0, target_labels, 0)

        target_boxes = img_labels[:, :, 1:].view(-1, 4)[max_pos_max]

        norm_metric = ((iou_max * metric) / (metric_max + self.eps)).amax(1).unsqueeze(2)
        target_scores = target_labels * norm_metric

        targets = torch.cat([target_scores, target_boxes], dim=2)

        return targets, mask_pos_sum

    def __call__(self, labels, preds, gpoints, gstrides, img_size):
        # labels: B、T、1 + 4
        # preds: B、A、num_cls + 4
        # cpoints: A、2
        # cstrides: A、1
        # mask: B、T、1

        targets, mask_pos = self.build_targets(labels, preds, gpoints, gstrides, img_size)
        print(targets)



labels = torch.tensor([
    [0, 0.0900, 0.0900, 1.8900, 1.8900],
    [1, 1.0950, 1.0950, 2.6850, 2.6850],
    [0, 0.1050, 1.0950, 0.9150, 1.9050],
    [1, 1.2000, 0.5850, 2.7000, 1.6950],
    [2, 0.0900, 2.0850, 2.9100, 2.8950]])

preds = torch.tensor([
    [[0.10, 0.20, 0.30, 0.05, 0.05, 1.05, 1.05],
     [0.15, 0.25, 0.35, 1.05, 0.05, 2.05, 1.05],
     [0.20, 0.30, 0.40, 2.05, 0.05, 3.05, 1.05],
     [0.25, 0.35, 0.45, 0.05, 1.05, 1.05, 2.05],
     [0.30, 0.40, 0.50, 1.05, 1.05, 2.05, 2.05],
     [0.35, 0.45, 0.55, 2.05, 1.05, 3.05, 2.05],
     [0.40, 0.50, 0.60, 0.05, 2.05, 1.05, 3.05],
     [0.45, 0.55, 0.65, 1.05, 2.05, 2.05, 3.05],
     [0.50, 0.60, 0.70, 2.05, 2.05, 3.05, 3.05]],

    [[0.10, 0.20, 0.30, 0.05, 0.05, 1.05, 1.05],
     [0.15, 0.25, 0.35, 1.05, 0.05, 2.05, 1.05],
     [0.20, 0.30, 0.40, 2.05, 0.05, 3.05, 1.05],
     [0.25, 0.35, 0.45, 0.05, 1.05, 1.05, 2.05],
     [0.30, 0.40, 0.50, 1.05, 1.05, 2.05, 2.05],
     [0.35, 0.45, 0.55, 2.05, 1.05, 3.05, 2.05],
     [0.40, 0.50, 0.60, 0.05, 2.05, 1.05, 3.05],
     [0.45, 0.55, 0.65, 1.05, 2.05, 2.05, 3.05],
     [0.50, 0.60, 0.70, 2.05, 2.05, 3.05, 3.05]]])

# grid
gpoints = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5]])
gstrides = torch.tensor([[8], [8], [8], [8], [8], [8], [8], [8], [8]])
batch_size = 2
img_size = torch.tensor([24, 24])

loss = Loss(1, 1, 3, 'cpu')
loss(labels, preds, gpoints, gstrides, img_size)
