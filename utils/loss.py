import torch
from torch import nn
import torch.nn.functional as F
from box import bbox_iou, bbox2dist


class Loss:
    def __init__(self, alpha, beta, topk, device, eps=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
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

        mask = torch.gt(out.sum(2, keepdim=True), 0).to(torch.int64)

        return out, mask
