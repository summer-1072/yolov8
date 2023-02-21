import torch
import torch.nn.functional as F
from box import xywh2xyxy, bbox_iou


class Loss:
    def __init__(self, alpha, beta, topk, device):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.device = device

    def format_targets(self, targets, batch_size, scale):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            indices = targets[:, 0]
            index, count = indices.unique(return_counts=True)
            out = torch.zeros(batch_size, count.max(), 5, device=self.device)
            for index in range(batch_size):
                matches = indices == index
                num = matches.sum()
                if num:
                    out[index, :num] = targets[matches, 1:]

            out[:, :, 1:5] = xywh2xyxy(out[:, :, 1:5] * scale)

        return out

    def build_targets(self, preds, targets, cpoints, mask, eps=1e-5):
        # preds: B、A、num_cls + 4
        # targets: B、num_box、1 + 4
        # cpoints: A、2
        # mask: B、num_box、1

        batch_size, box_size = targets.shape[:2]
        anchor_size = preds.shape[1]

        # compute metric
        ind = torch.zeros([2, batch_size, box_size], dtype=torch.long)
        ind[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, box_size)
        ind[1] = targets[:, :, 0]
        cls = preds[:, :, 0:-4][ind[0], :, ind[1]]
        iou = bbox_iou(targets[:, :, 1:], preds[:, :, -4:], True, 'CIoU').squeeze(3).clamp(0)
        metric = cls.pow(self.alpha) * iou.pow(self.beta)

        # match box
        point_size = cpoints.shape[0]
        left_top, right_bottom = targets[:, :, 1:].view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((cpoints.unsqueeze(0) - left_top, right_bottom - cpoints.unsqueeze(0)), 2).view(
            batch_size, box_size, point_size, 4)
        match_index = deltas.amin(3).gt_(eps)

        # top box
        match_metric = metric * match_index
        top_metric, top_index = torch.topk(match_metric, self.topk, 2)
        if len(mask):
            top_mask = mask.repeat([1, 1, self.topk]).bool()
        else:
            top_mask = (top_metric.max(2, keepdim=True).values > eps).repeat([1, 1, self.topk])

        top_index = torch.where(top_mask, top_index, 0)
        is_in_topk = F.one_hot(top_index, anchor_size).sum(2)
        top_index = torch.where(is_in_topk > 1, 0, is_in_topk)

        # candidate box
        index = match_index * top_index * mask

        # check repeat box
        repeat_index = index.sum(1)
        if repeat_index.max() > 1:
            repeat_index = (repeat_index.unsqueeze(1) > 1).repeat([1, box_size, 1])
            iou_max_index = iou.argmax(1)
            iou_max_index = F.one_hot(iou_max_index, box_size)
            iou_max_index = iou_max_index.permute(0, 2, 1).to(iou.dtype)
            index = torch.where(repeat_index, iou_max_index, index)

        fg_mask = index.sum(1)
        target_gt_idx = index.argmax(1)

        print(index)
        print(fg_mask)
        print(target_gt_idx)

        ind = torch.arange(end=batch_size, dtype=torch.int64).unsqueeze(1)
        target_gt_idx = target_gt_idx + ind * 2
        print(target_gt_idx)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        return iou, metric, index

    def __call__(self, preds, targets, cpoints, masks):
        targets = self.format_targets(targets, len(preds), scale)
        iou, metric, index = self.build_targets(preds, targets, cpoints, masks)


# targets = torch.tensor([
#     [[0, 0.1, 0.1, 1.9, 1.9], [1, 1.1, 1.1, 2.7, 2.7]],
#     [[0, 1.2, 0.8, 2.7, 1.7], [0., 0., 0., 0., 0.]]
# ])

targets = torch.tensor([
    [0, 0, 0.33, 0.33, 0.6, 0.6],
    [0, 1, 0.63, 0.63, 0.53, 0.53],
    [1, 2, 0.65, 0.38, 0.5, 0.37]])

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

# true mask
masks = torch.tensor([[[1], [1]], [[1], [0]]])

# grid
cpoints = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5]])

batch_size = 2
scale = torch.tensor([3, 3, 3, 3])

loss = Loss(1, 1, 4, 'cpu')
loss(preds, targets, cpoints, masks)
