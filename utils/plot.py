import cv2
import math
import torch
import numpy as np
from util import color
from box import scale_offset
from collections import Counter
import matplotlib.pyplot as plt


def plot_images(imgs, labels, save_file, max_shape=(2880, 3840), max_subplots=16):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().float().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().float().numpy()

    if np.max(imgs[0]) <= 1:
        imgs *= 255.0

    B, C, H, W = imgs.shape
    B = min(B, max_subplots)
    num = np.ceil(B ** 0.5)

    labels[:, 2:6] = scale_offset(labels[:, 2:6],  W, H)

    img_mosaic = np.full((int(num * H), int(num * W), 3), 0, dtype=np.uint8)
    for index, img in enumerate(imgs):
        if index == max_subplots:
            break

        x, y = int(W * (index // num)), int(H * (index % num))
        img = img[::-1].transpose(1, 2, 0)
        img = np.ascontiguousarray(img)
        img_mosaic[y:y + H, x:x + W, :] = img
        cv2.rectangle(img_mosaic, (x, y), (x + W, y + H), (255, 255, 255), 2)

        img_labels = labels[labels[:, 0] == index]
        for img_label in img_labels:
            pt1 = (round(img_label[2]) + x, round(img_label[3]) + y)
            pt2 = (round(img_label[4]) + x, round(img_label[5]) + y)
            cv2.rectangle(img_mosaic, pt1, pt2, color(int(img_label[1])), 2)

    ratio = min(max_shape[0] / num / H, max_shape[1] / num / W)
    if ratio < 1:
        H = math.ceil(ratio * H)
        W = math.ceil(ratio * W)
        img_mosaic = cv2.resize(img_mosaic, tuple(int(x * num) for x in (W, H)), cv2.INTER_LINEAR)

    cv2.imwrite(save_file, img_mosaic)


def plot_labels(labels, cls, save_file):
    indices = []
    centers = []
    whs = []

    for label in labels:
        indices.extend([int(x) for x in label[:, 0].tolist()])
        centers.extend(label[:, 1:3].tolist())
        whs.extend(label[:, 3:5].tolist())

    count = dict(Counter(indices))
    t_indices = sorted(count.items(), key=lambda x: x[0])
    t_data = [[cls[x[0]], x[1]] for x in t_indices]

    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    plt.title('target', fontsize=12)
    for k, v in t_data:
        bar = plt.bar(k, v, log=True)
        plt.bar_label(bar, fontsize=8)

    plt.subplot(2, 2, 3)
    plt.title('center', fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter([x[0] for x in centers], [x[1] for x in centers], s=0.05)

    plt.subplot(2, 2, 4)
    plt.title('wh', fontsize=12)
    plt.xlabel('w')
    plt.ylabel('h')
    plt.scatter([x[0] for x in whs], [x[1] for x in whs], s=0.05)

    plt.savefig(save_file)
