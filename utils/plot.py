import cv2
import math
import torch
import numpy as np
from util import color
import matplotlib.pyplot as plt


def plot_images(imgs, labels, file_path, max_shape=(2880, 3840), max_subplots=16):
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().float().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().float().numpy()

    if np.max(imgs[0]) <= 1:
        imgs *= 255.0

    B, C, H, W = imgs.shape
    B = min(B, max_subplots)
    num = np.ceil(B ** 0.5)

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

    cv2.imwrite(file_path, img_mosaic)


def plot_labels(train_labels, val_labels, cls, file_path):
    train_labels = np.concatenate(train_labels, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    train_cls, train_count = np.unique(train_labels[:, 0], return_counts=True)
    val_cls, val_count = np.unique(val_labels[:, 0], return_counts=True)

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.25)

    plt.subplot(3, 1, 1)
    plt.title('train cls', fontsize=12)
    for i in range(len(train_cls)):
        bar = plt.bar(cls[int(train_cls[i])], train_count[int(train_cls[i])], log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(3, 1, 2)
    plt.title('val cls', fontsize=12)
    for i in range(len(val_cls)):
        bar = plt.bar(cls[int(val_cls[i])], val_count[int(val_cls[i])], log=True)
        plt.bar_label(bar, fontsize=6)

    plt.subplot(3, 2, 5)
    plt.title('train box', fontsize=12)
    plt.xlabel('w')
    plt.ylabel('h')
    plt.scatter(train_labels[:, 3] - train_labels[:, 1], train_labels[:, 4] - train_labels[:, 2], s=0.5)

    plt.subplot(3, 2, 6)
    plt.title('val box', fontsize=12)
    plt.scatter(val_labels[:, 3] - val_labels[:, 1], val_labels[:, 4] - val_labels[:, 2], s=0.5)
    plt.xlabel('w')
    plt.ylabel('h')

    plt.savefig(file_path)


def plot_metrics():
    pass
