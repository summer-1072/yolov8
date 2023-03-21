import numpy as np


def build_metric(matrix, conf, pred_cls, target_cls, eps=1e-16):
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
    index = np.convolve(np.concatenate((p * mean[0], mean, p * mean[-1]), 0), np.ones(num) / num, mode='valid').argmax()

    return P[:, index], R[:, index], F1[:, index], AP
