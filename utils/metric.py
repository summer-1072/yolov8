import numpy as np


def build_metric(matrix, conf, pred_cls, target_cls, eps=1e-16):
    index = np.argsort(-conf)
    matrix, conf, pred_cls = matrix[index], conf[index], pred_cls[index]

    cls, nt = np.unique(target_cls, return_counts=True)
