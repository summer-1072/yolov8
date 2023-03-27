import torch
import numpy as np
from box import rescale_box, bbox_iou


def smooth(x, f=0.05):
    nf = round(len(x) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * x[0], x, p * x[-1]), 0)

    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


class Metric:
    def __init__(self, device):
        self.device = device
        self.num_img = 0
        self.status = []
        self.iouv = torch.linspace(0.5, 0.95, 10)

    def update_status(self, labels, preds, img_sizes):
        for index, pred in enumerate(preds):
            img_size = img_sizes[index]
            label = labels[labels[:, 0] == index]

            matrix = torch.zeros(pred.shape[0], self.iouv.shape[0], dtype=torch.bool, device=self.device)
            self.num_img += 1
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

    def build_metrics(self, eps=1e-16):
        matrix, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.status)]

        index = np.argsort(-conf)
        matrix, conf, pred_cls = matrix[index], conf[index], pred_cls[index]

        cls, count = np.unique(target_cls, return_counts=True)  # sorted

        num_cls, num_ins = len(cls), sum(count)

        P, R, AP = np.zeros((num_cls, 1000)), np.zeros((num_cls, 1000)), np.zeros((num_cls, matrix.shape[1]))

        for i in range(cls.shape[0]):
            matches = pred_cls == cls[i]
            if count[i] == 0 or matches.sum() == 0:
                continue

            TP = matrix[matches].cumsum(0)
            FP = (1 - matrix[matches]).cumsum(0)

            # Precision
            precision = TP / (TP + FP)
            P[i] = np.interp(-np.linspace(0, 1, 1000), -conf[matches], precision[:, 0], left=1)

            # Recall
            recall = TP / (count[i] + eps)
            R[i] = np.interp(-np.linspace(0, 1, 1000), -conf[matches], recall[:, 0], left=0)

            # AP
            for j in range(matrix.shape[1]):
                p = np.concatenate(([1.0], precision[:, j], [0.0]))
                p = np.flip(np.maximum.accumulate(np.flip(p)))
                r = np.concatenate(([0.0], recall[:, j], [1.0]))
                AP[i, j] = np.trapz(np.interp(np.linspace(0, 1, 101), r, p), np.linspace(0, 1, 101))

        # F1
        F1 = 2 * P * R / (P + R + eps)

        # F1 max index
        index = smooth(F1.mean(0), 0.1).argmax()
        P, R, F1 = P[:, index], R[:, index], F1[:, index]

        # metrics
        precision = round(P.mean(), 5) if len(P) else 0.0
        recall = round(R.mean(), 5) if len(R) else 0.0
        mAP50 = round(AP[:, 0].mean(), 5) if len(AP) else 0.0
        mAP75 = round(AP[:, 5].mean(), 5) if len(AP) else 0.0
        mAP50_95 = round(AP.mean(), 5) if len(AP) else 0.0

        print(('%22s' + '%12s' * 7) % ('images', 'class', 'instances', 'P', 'R', 'mAP50', 'mAP75', 'mAP50-95'))
        print(('%22s' + '%12s' * 7) % (self.num_img, num_cls, num_ins, precision, recall, mAP50, mAP75, mAP50_95))

        weight = [0.0, 0.0, 0.1, 0.9]  # [P, R, mAP@0.5, mAP@0.5:0.95]
        fitness = (np.array([precision, recall, mAP50, mAP50_95]) * weight).sum()

        return precision, recall, mAP50, mAP75, mAP50_95, fitness
