import cv2
import math
import torch
import torchvision
import numpy as np


# B、A、C or A、C
def gap2box(gap, grid, dim=-1):
    left, top, right, bottom = gap.chunk(4, dim)
    x, y = grid.chunk(2, dim)

    return torch.cat((x - left, y - top, x + right, y + bottom), dim)


def box2gap(box, grid, reg_max, dim=-1):
    left, top, right, bottom = box.chunk(4, dim)
    x, y = grid.chunk(2, dim)

    return torch.cat((x - left, y - top, right - x, bottom - y), dim).clamp(0, reg_max - 1.01)


def bbox_iou(box1, box2, type='IoU', eps=1e-5, dim=-1):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection Area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if type == 'GIoU' or type == 'DIoU' or type == 'CIoU':
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        d2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        c2 = cw ** 2 + ch ** 2 + eps

        if type == 'GIoU':
            c_area = cw * ch + eps
            iou = iou - (c_area - union) / c_area

        elif type == 'DIoU':
            iou = iou - d2 / c2

        elif type == 'CIoU':
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            alpha = v / (1 - iou + v + eps)

            iou = iou - (d2 / c2 + alpha * v)

    return iou


def letterbox(img, new_shape, stride):
    shape = img.shape[:2]
    ratio = min(min(new_shape[0] / shape[0], new_shape[1] / shape[1]), 1)

    unpad_shape = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dx, dy = ((new_shape[1] - unpad_shape[0]) % stride) / 2, ((new_shape[0] - unpad_shape[1]) % stride) / 2

    if shape[::-1] != unpad_shape:
        img = cv2.resize(img, unpad_shape, cv2.INTER_LINEAR)

    top, bottom = int(round(dy - 0.1)), int(round(dy + 0.1))
    left, right = int(round(dx - 0.1)), int(round(dx + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, img.shape[:2], (dy, dx)


def scale_box(box, h, w, dy, dx, dim=-1):
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    return np.stack((w * x1 + dx, h * y1 + dy, w * x2 + dx, h * y2 + dy), dim)


def rescale_box(shape0, shape1, box):
    gain = min(shape1[0] / shape0[0], shape1[1] / shape0[1])
    pad = (shape1[1] - shape0[1] * gain) / 2, (shape1[0] - shape0[0] * gain) / 2

    box[:, [0, 2]] -= pad[0]  # x padding
    box[:, [1, 3]] -= pad[1]  # y padding
    box[:, :4] /= gain

    box[:, 0].clamp_(0, shape0[1])
    box[:, 1].clamp_(0, shape0[0])
    box[:, 2].clamp_(0, shape0[1])
    box[:, 3].clamp_(0, shape0[0])

    return box


def non_max_suppression(preds, conf_t, multi_label, max_box, max_wh, iou_t, max_det, merge):
    B = preds.shape[0]
    num_cls = preds.shape[2] - 4
    candidates = preds[:, :, 4:4 + num_cls].amax(2) > conf_t
    multi_label &= (num_cls > 1)

    output = [torch.zeros((0, 6), device=preds.device)] * B

    for index, pred in enumerate(preds):
        pred = pred[candidates[index]]

        if pred.shape[0] == 0:
            continue

        box, cls = pred.split((4, num_cls), 1)

        if multi_label:
            i, j = (cls > conf_t).nonzero(as_tuple=False).T
            pred = torch.cat((box[i], pred[i, j + 4, None], j[:, None].float()), 1)

        else:
            conf, j = cls.max(1, keepdim=True)
            pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_t]

        if pred.shape[0] == 0:
            continue

        pred = pred[pred[:, 4].argsort(descending=True)[:max_box]]

        scale = pred[:, 5:6] * max_wh
        boxes, confs = pred[:, :4] + scale, pred[:, 4]
        indices = torchvision.ops.nms(boxes, confs, iou_t)
        indices = indices[:max_det]

        if merge:
            iou = bbox_iou(boxes[indices].unsqueeze(1), boxes.unsqueeze(0)) > iou_t
            weights = iou.squeeze(2) * confs.unsqueeze(0)
            pred[indices, :4] = torch.mm(weights, pred[:, :4]).float() / weights.sum(1, keepdim=True)

        output[index] = pred[indices]

    return output
