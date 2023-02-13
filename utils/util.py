import time
import torch
import numpy as np


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()


def one_hot(label, num):
    matrix = np.diag([1 for _ in range(num)])
    label = np.vectorize(lambda i: matrix[i], signature='()->(n)')(label)

    return label


def color(index):
    hex = ['#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#FF3838', '#FF9D97',
           '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7']

    c = hex[index % len(hex)]
    c = tuple(int(c[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return (c[2], c[1], c[0])
