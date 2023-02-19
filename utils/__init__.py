import torch
import math
import numpy as np
import torch.nn.functional as F


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


preds = torch.tensor([
    [[0.10, 0.20, 0.30, 0.05, 0.05, 1.05, 1.05],
     [0.15, 0.25, 0.35, 1.05, 0.05, 2.05, 1.05],
     [0.20, 0.30, 0.40, 2.05, 0.05, 3.05, 1.05],
     [0.25, 0.35, 0.45, 0.05, 1.05, 1.05, 2.05],
     [0.30, 0.40, 0.50, 1.05, 1.05, 2.05, 2.05],
     [0.35, 0.45, 0.55, 2.05, 1.05, 3.05, 2.05],
     [0.40, 0.50, 0.60, 0.05, 2.05, 1.05, 3.05],
     [0.45, 0.55, 0.65, 1.05, 2.05, 2.05, 3.05],
     [0.50, 0.60, 0.70, 2.05, 2.05, 3.05, 3.05]]
])

targets = torch.tensor([
    [[0, 0.1, 0.9, 0.8, 1.6], [1, 1.2, 1.4, 2.2, 2.6]]
])

pd_scores, pd_bboxes = preds.split((3, 4), 2)
gt_labels, gt_bboxes = targets.split((1, 4), 2)

# true mask
mask_gt = torch.tensor([[[1],
                         [1]]])

# grid
anc_points = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                           [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                           [0.5, 2.5], [1.5, 2.5], [2.5, 2.5]
                           ])

ind = torch.zeros([2, 1, 2], dtype=torch.long)
ind[0] = torch.arange(end=1).view(-1, 1).repeat(1, 2)
ind[1] = gt_labels.long().squeeze(-1)
bbox_scores = pd_scores[ind[0], :, ind[1]]
overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
align_metric = bbox_scores.pow(1) * overlaps.pow(1)

n_anchors = anc_points.shape[0]
bs, n_boxes, _ = gt_bboxes.shape
lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
mask_in_gts = bbox_deltas.amin(3).gt_(1e-9)

topk_mask = mask_gt.repeat([1, 1, 8]).bool()

num_anchors = align_metric.shape[-1]  # h*w
# (b, max_num_obj, topk)
topk_metrics, topk_idxs = torch.topk(align_metric, 8, dim=-1, largest=True)
if topk_mask is None:
    topk_mask = (topk_metrics.max(-1, keepdim=True) > 1e-9).tile([1, 1, 8])
# (b, max_num_obj, topk)
topk_idxs = torch.where(topk_mask, topk_idxs, 0)
# (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
# filter invalid bboxes
is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)

mask_pos = is_in_topk * mask_in_gts * mask_gt

print(mask_pos)
print(align_metric)
print(overlaps)

# print(mask_gt.repeat([1, 1, 10]).bool())
