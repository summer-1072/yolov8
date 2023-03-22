import torch
import numpy as np
from box import rescale_box, bbox_iou


class Metric:
    def __init__(self, iouv):
        self.iouv = iouv
        self.status = []

    def update_status(self, labels, preds, img_sizes):
        for index, pred in enumerate(preds):
            label = labels[labels[:, 0] == index]
            size = img_sizes[index]

            pred[:, :4] = rescale_box(size[0], size[1], pred[:, :4])
            label[:, 2:] = rescale_box(size[0], size[1], label[:, 2:])

            iou = bbox_iou(label[:, 2:].unsqueeze(1), pred[:, :4].unsqueeze(0), 'IoU').squeeze(2).clamp(0)
            cls = label[:, 1:2] == pred[:, 5]
            matrix = np.zeros((pred.shape[0], self.iouv.shape[0])).astype(bool)

            for i in range(len(self.iouv)):
                y, x = torch.where((iou >= self.iouv[i]) & cls)
                if len(x):
                    matches = torch.cat((torch.stack((y, x), 1), iou[y, x][:, None]), 1).cpu().numpy()
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                    matrix[matches[:, 1].astype(int), i] = True

            self.status.append((torch.tensor(matrix), pred[:, 4], pred[:, 5], label[:, 1]))

    def calculate_metric(self, eps=1e-16):
        status = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.status)]
        matrix, conf, pred_cls, target_cls = status[0], status[1], status[2], status[3]

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
        mean = F1.mean(0)
        num = round(len(mean) * 0.1 * 2) // 2 + 1
        p = np.ones(num // 2)
        index = np.convolve(np.concatenate((p * mean[0], mean, p * mean[-1]), 0), np.ones(num) / num,
                            mode='valid').argmax()

        return P[:, index], R[:, index], F1[:, index], AP, unique_cls.astype(int)
