import torch
import numpy as np
from box import rescale_box, bbox_iou


def smooth(x, f=0.05):
    nf = round(len(x) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * x[0], x, p * x[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


class Metric:
    def __init__(self, iouv, device):
        self.iouv = iouv
        self.device = device
        self.status = []
        self.indices = {}

    def update_status(self, labels, preds, img_sizes):
        for index, pred in enumerate(preds):
            img_size = img_sizes[index]
            label = labels[labels[:, 0] == index]

            matrix = torch.zeros(pred.shape[0], self.iouv.shape[0], dtype=torch.bool, device=self.device)

            if label.shape[0] != 0:
                if pred.shape[0] == 0:
                    self.status.append((matrix, *torch.zeros((2, 0), device=self.device), label[:, 1]))

                else:
                    pred[:, :4] = rescale_box(img_size[0], img_size[1], pred[:, :4])
                    label[:, 2:] = rescale_box(img_size[0], img_size[1], label[:, 2:])

                    iou = bbox_iou(label[:, 2:].unsqueeze(1), pred[:, :4].unsqueeze(0), 'IoU').squeeze(2).clamp(0)
                    cls = label[:, 1:2] == pred[:, 5]

                    for i in range(len(self.iouv)):
                        y, x = torch.where((iou >= self.iouv[i]) & cls)
                        if len(x):
                            matches = torch.cat((torch.stack((y, x), 1), iou[y, x][:, None]), 1).cpu().numpy()
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                            matrix[matches[:, 1].astype(int), i] = True

                    self.status.append((matrix, pred[:, 4], pred[:, 5], label[:, 1]))

    def build_indices(self, eps=1e-16):
        matrix, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.status)]

        index = np.argsort(-conf)
        matrix, conf, pred_cls = matrix[index], conf[index], pred_cls[index]

        unique_cls, count = np.unique(target_cls, return_counts=True)
        num_cls = unique_cls.shape[0]
        P, R, AP = np.zeros((num_cls, 1000)), np.zeros((num_cls, 1000)), np.zeros((num_cls, matrix.shape[1]))

        for index, cls in enumerate(unique_cls):
            matches = pred_cls == cls
            cls_count = count[index]
            if cls_count == 0 or matches.sum() == 0:
                continue

            TP = matrix[matches].cumsum(0)
            FP = (1 - matrix[matches]).cumsum(0)

            # Precision
            precision = TP / (TP + FP)
            P[index] = np.interp(-np.linspace(0, 1, 1000), -conf[matches], precision[:, 0], left=1)

            # Recall
            recall = TP / (cls_count + eps)
            R[index] = np.interp(-np.linspace(0, 1, 1000), -conf[matches], recall[:, 0], left=0)

            # AP
            for i in range(matrix.shape[1]):
                p = np.concatenate(([1.0], precision[:, i], [0.0]))
                p = np.flip(np.maximum.accumulate(np.flip(p)))
                r = np.concatenate(([0.0], recall[:, i], [1.0]))
                AP[index, i] = np.trapz(np.interp(np.linspace(0, 1, 101), r, p), np.linspace(0, 1, 101))

        # F1
        F1 = 2 * P * R / (P + R + eps)

        # F1 max
        index = smooth(F1.mean(0), f=0.1)

        self.indices = {'P': P[:, index], 'R': R[:, index], 'F1': F1[:, index], 'AP': AP, 'cls': unique_cls.astype(int)}

    def build_results(self):
        precision = self.indices['P'].mean() if len(self.indices['P']) else 0.0
        recall = self.indices['R'].mean() if len(self.indices['R']) else 0.0
        mAP50 = self.indices['AP'][:, 0].mean() if len(self.indices['AP']) else 0.0
        mAP75 = self.indices['AP'][:, 5].mean() if len(self.indices['AP']) else 0.0
        mAP50_95 = self.indices['AP'].mean() if len(self.indices['AP']) else 0.0
        weight = [0.0, 0.0, 0.1, 0.9]  # P、R、mAP@0.5、mAP@0.5:0.95
        fitness = (np.array([precision, recall, mAP50, mAP50_95]) * weight).sum()

        return {'precision': precision, 'recall': recall, 'mAP50': mAP50, 'mAP50_95': mAP50_95, 'fitness': fitness}
