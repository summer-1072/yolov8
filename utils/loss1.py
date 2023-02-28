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

    def build_targets(self, labels, preds, gpoints, gstrides, img_size):
        labels, mask = self.format_labels(labels, preds)

        scale_labels = labels[:, :, 1:5] * img_size[[1, 0, 1, 0]]
        scale_preds = torch.cat([preds[:, :, :-4], preds[:, :, -4:] * gstrides], 2)
        scale_gpoints = gpoints * gstrides

        cls, iou, metric = self.build_metrics(scale_labels, scale_preds)

        print(cls)

    def __call__(self, labels, preds, gpoints, gstrides, img_size):
        # labels: B、T、1 + 4
        # preds: B、A、num_cls + 4
        # cpoints: A、2
        # cstrides: A、1
        # mask: B、T、1

        self.build_targets(labels, preds, gpoints, gstrides, img_size)

        pass


labels = torch.tensor([
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

# grid
gpoints = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5]])
gstrides = torch.tensor([[8], [8], [8], [8], [8], [8], [8], [8], [8]])
batch_size = 2
img_size = torch.tensor([24, 24])

loss = Loss(1, 1, 3, 'cpu')
loss(labels, preds, gpoints, gstrides, img_size)
