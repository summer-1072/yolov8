import torch
import torch.nn.functional as F
from box import xywh2xyxy, bbox_iou


class Loss:
    def __init__(self, alpha, beta, topk, num_cls, device):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.num_cls = num_cls
        self.device = device

    def format_labels(self, labels, preds, scale):
        batch_size = preds.shape[0]

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

            out[:, :, 1:5] = xywh2xyxy(out[:, :, 1:5] * scale)

        return out

    def search_positives(self, preds, labels, cpoints, mask, num_cls, eps=1e-5):
        batch_size, box_size = labels.shape[:2]
        anchor_size = preds.shape[1]

        # compute metric
        index = torch.zeros([2, batch_size, box_size], dtype=torch.long)
        index[0] = torch.arange(batch_size).view(-1, 1).repeat(1, box_size)
        index[1] = labels[:, :, 0]
        cls = preds[:, :, 0:num_cls][index[0], :, index[1]]
        iou = bbox_iou(labels[:, :, 1:5].unsqueeze(2), preds[:, :, num_cls:].unsqueeze(1), 'CIoU').squeeze(3).clamp(0)
        metric = cls.pow(self.alpha) * iou.pow(self.beta)

        # match anchor
        left_top, right_bottom = labels[:, :, 1:5].view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((cpoints.unsqueeze(0) - left_top, right_bottom - cpoints.unsqueeze(0)), 2).view(
            batch_size, box_size, anchor_size, 4)
        match_indices = deltas.amin(3).gt_(eps)

        # top anchor
        match_metric = metric * match_indices
        top_metric, top_indices = torch.topk(match_metric, self.topk, 2)
        if len(mask):
            top_mask = mask.repeat([1, 1, self.topk]).bool()
        else:
            top_mask = (top_metric.max(2, keepdim=True).values > eps).repeat([1, 1, self.topk])

        top_indices = torch.where(top_mask, top_indices, 0)
        is_in_topk = F.one_hot(top_indices, anchor_size).sum(2)
        top_indices = torch.where(is_in_topk > 1, 0, is_in_topk)

        # candidate anchor
        pos_indices = match_indices * top_indices * mask

        # duplicate anchor
        repeat_pos_indices = pos_indices.sum(1)
        if repeat_pos_indices.max() > 1:
            repeat_pos_indices = (repeat_pos_indices.unsqueeze(1) > 1).repeat([1, box_size, 1])
            iou_max_indices = iou.argmax(1)
            iou_max_indices = F.one_hot(iou_max_indices, box_size)
            iou_max_indices = iou_max_indices.permute(0, 2, 1).to(iou.dtype)
            pos_indices = torch.where(repeat_pos_indices, iou_max_indices, pos_indices)

        return iou, metric, pos_indices

    # def build_targets(self, iou, metric, mask_pos, labels):
    #     mask_pos = index.sum(1)
    #     max_pos = mask_pos.argmax(1)
    #     ind = torch.arange(batch_size, dtype=torch.int64, device=self.device).unsqueeze(1)
    #
    #     max_pos = max_pos + ind * box_size
    #     cls = labels[:, :, 0].long().flatten()[max_pos]
    #     cls.clamp(0)
    #
    #     box = labels[:, :, 1:].view(-1, 4)[max_pos]
    #     target_scores = F.one_hot(cls, 3)
    #     print(target_scores)
    #     fg_scores_mask = mask_pos[:, :, None].repeat(1, 1, 3)
    #     print(fg_scores_mask)
    #
    #     target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
    #     print(target_scores)

    def __call__(self, preds, labels, cpoints, masks):
        # preds: B、A、num_cls + 4
        # labels: B、num_box、1 + 4
        # cpoints: A、2
        # mask: B、num_box、1

        labels = self.format_labels(labels, preds, scale)
        iou, metric, pos_indices = self.search_positives(preds, labels, cpoints, masks, self.num_cls)

        print(pos_indices)


# targets = torch.tensor([
#     [[0, 0.0900, 0.0900, 1.8900, 1.8900], [1, 1.0950, 1.0950, 2.6850, 2.6850], [0., 0., 0., 0., 0.]],
#     [[0, 0.09,1.0950, 0.9, 1.89], [1, 1.2000, 0.5850, 2.7000, 1.6950], [2., 0.09, 2.1, 2.9, 2.9]]
# ])

targets = torch.tensor([
    [0, 0, 0.33, 0.33, 0.6, 0.6],
    [0, 1, 0.63, 0.63, 0.53, 0.53],
    [1, 0, 0.17, 0.5, 0.27, 0.27],
    [1, 1, 0.65, 0.38, 0.5, 0.37],
    [1, 2, 0.5, 0.83, 0.94, 0.27]])

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
masks = torch.tensor([[[1], [1], [0]], [[1], [1], [1]]])

# grid
cpoints = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5]])

batch_size = 2
scale = torch.tensor([3, 3, 3, 3])

loss = Loss(1, 1, 3, 3, 'cpu')
loss(preds, targets, cpoints, masks)
