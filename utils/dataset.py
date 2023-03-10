import cv2
import json
import math
import torch
import random
import imagesize
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from box import scale_offset, unscale_offset, letterbox


def build_labels(input_file, output_file, image_dir, cls):
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for index in tqdm(range(len(lines)), desc=f'reading {input_file}, {len(lines)} records'):
            name = lines[index]['name']
            labels = lines[index]['labels']
            img_w, img_h = imagesize.get(image_dir + '/' + name)

            targets = []
            for label in labels:
                category = label['category']
                if category in cls:
                    c = cls.index(category)
                    box2d = label['box2d']
                    x1 = round(box2d['x1'] / img_w, 4)
                    y1 = round(box2d['y1'] / img_h, 4)
                    x2 = round(box2d['x2'] / img_w, 4)
                    y2 = round(box2d['y2'] / img_h, 4)

                    targets.append(','.join([str(i) for i in [c, x1, y1, x2, y2]]))

            records.append(name + '  ' + '  '.join(targets))

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(record + '\n')


def read_labels(file):
    indices, imgs, labels = [], [], []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        print('reading {}, {} coco'.format(file, len(lines)))
        for index in range(len(lines)):
            indices.append(index)
            line = lines[index].replace('\n', '').split('  ')
            imgs.append(line[0])
            labels.append(np.array([[float(i) for i in obj.split(',')] for obj in line[1:]]))

    return indices, imgs, labels


def load_image(file, size):
    img = cv2.imread(file)
    h, w = img.shape[:2]
    ratio = size / max(h, w)
    if ratio != 1:
        img = cv2.resize(img, (math.ceil(w * ratio), math.ceil(h * ratio)), cv2.INTER_LINEAR)

    return img, img.shape[:2]


def mosaic(index, indices, img_dir, imgs, labels, new_shape):
    c_x = int(random.uniform(new_shape[1] // 2, 2 * new_shape[1] - new_shape[1] // 2))
    c_y = int(random.uniform(new_shape[0] // 2, 2 * new_shape[0] - new_shape[0] // 2))

    t_indices = [index] + random.choices(indices, k=3)
    random.shuffle(t_indices)

    img4 = np.full((new_shape[0] * 2, new_shape[1] * 2, 3), 114, dtype=np.uint8)
    img4_labels = []
    for (i, index) in enumerate(t_indices):
        img_file = img_dir + '/' + imgs[index]

        img, (h, w) = load_image(img_file, max(new_shape))

        if i == 0:
            x1a, y1a, x2a, y2a = max(c_x - w, 0), max(c_y - h, 0), c_x, c_y
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = c_x, max(c_y - h, 0), min(c_x + w, new_shape[1] * 2), c_y
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(c_x - w, 0), c_y, c_x, min(new_shape[0] * 2, c_y + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = c_x, c_y, min(c_x + w, new_shape[1] * 2), min(new_shape[0] * 2, c_y + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # labels
        offset_x, offset_y = x1a - x1b, y1a - y1b
        img_labels = labels[index].copy()
        if len(img_labels):
            img_labels[:, 1:5] = scale_offset(img_labels[:, 1:5], w, h, offset_x, offset_y)
            img4_labels.append(img_labels)

    img4_labels = np.concatenate(img4_labels, 0)

    img4_labels[:, [1, 3]] = np.clip(img4_labels[:, [1, 3]], 0, 2 * new_shape[1])
    img4_labels[:, [2, 4]] = np.clip(img4_labels[:, [2, 4]], 0, 2 * new_shape[0])
    img4_labels = img4_labels / 2

    img4 = cv2.resize(img4, (new_shape[1], new_shape[0]), cv2.INTER_LINEAR)

    return img4, img4_labels


def affine_transform(img, labels, scale, translate):
    h, w = img.shape[:2]
    # center
    C = np.eye(3)
    C[0, 2] = -w / 2
    C[1, 2] = -h / 2

    # scale
    R = np.eye(3)
    scale = random.uniform(1.25 - scale, 1.25 + scale)
    R[0, 0], R[1, 1] = scale, scale

    # translate
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

    M = T @ R @ C

    img = cv2.warpAffine(img, M[:2], dsize=(w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    num = len(labels)
    if num:
        points = np.ones((num * 4, 3))
        points[:, :2] = labels[:, 1:5][:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(num * 4, 2)
        points = points @ M.T
        points = points[:, :2].reshape(num, 8)

        x = points[:, [0, 2, 4, 6]]
        y = points[:, [1, 3, 5, 7]]
        box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, num).T
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)

        labels[:, 1:5] = box

    return img, labels


def augment_hsv(img, h, s, v):
    if h or s or v:
        ratio = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=ratio.dtype)
        lut_hue = ((x * ratio[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * ratio[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * ratio[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def flip_up_down(img, labels):
    img = np.flipud(img)

    if len(labels):
        y1, y2 = labels[:, 2], labels[:, 4]
        labels[:, 2] = 1 - y2
        labels[:, 4] = 1 - y1

    return img, labels


def flip_left_right(img, labels):
    img = np.fliplr(img)

    if len(labels):
        x1, x2 = labels[:, 1], labels[:, 3]
        labels[:, 1] = 1 - x2
        labels[:, 3] = 1 - x1

    return img, labels


def check_labels(labels, box_t, wh_rt, eps=1e-3):
    if len(labels):
        w, h = labels[:, 3] - labels[:, 1], labels[:, 4] - labels[:, 2]
        wh_r = np.maximum(w / (h + eps), h / (w + eps))
        index = (w > box_t) & (h > box_t) & (wh_r < wh_rt)
        labels = labels[index]

    return labels


class LoadDataset(Dataset):
    def __init__(self, img_dir, label_file, hyp, augment):
        self.img_dir = img_dir
        self.hyp = hyp
        self.augment = augment
        self.indices, self.imgs, self.labels = read_labels(label_file)

    def __getitem__(self, index):
        if self.augment and self.hyp['mosaic']:
            img, labels = mosaic(index, self.indices, self.img_dir, self.imgs, self.labels, self.hyp['shape'])

        else:
            img_file = self.img_dir + '/' + self.imgs[index]
            img, (h, w) = load_image(img_file, max(self.hyp['shape']))
            img, ratio, (pad_w, pad_h) = letterbox(img, self.hyp['shape'], self.hyp['stride'])
            labels = self.labels[index].copy()
            if len(labels):
                labels[:, 1:5] = scale_offset(labels[:, 1:5], ratio * w, ratio * h, pad_w, pad_h)

        if self.augment and self.hyp['affine']:
            img, labels = affine_transform(img, labels, self.hyp['scale'], self.hyp['translate'])

        labels = check_labels(labels, self.hyp['box_t'], self.hyp['wh_rt'])

        num = len(labels)
        if num:
            labels[:, 1:5] = unscale_offset(labels[:, 1:5], img.shape[1], img.shape[0])

        if self.augment and random.random() < self.hyp['hsv']:
            augment_hsv(img, self.hyp['h'], self.hyp['s'], self.hyp['v'])

        if self.augment and random.random() < self.hyp['flipud']:
            img, labels = flip_up_down(img, labels)

        if self.augment and random.random() < self.hyp['fliplr']:
            img, labels = flip_left_right(img, labels)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        labels = torch.from_numpy(np.insert(labels, 0, 0, 1)) if num else torch.zeros((num, 6))

        return img, labels

    @staticmethod
    def collate_fn(batch):
        img, labels = zip(*batch)
        for index, label in enumerate(labels):
            label[:, 0] = index

        return torch.stack(img, 0), torch.cat(labels, 0)

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    import yaml

    cls = yaml.safe_load(open('../dataset/bdd100k/cls.yaml', encoding="utf-8"))
    build_labels('../dataset/bdd100k/labels/bdd100k_labels_images_train.json',
                 '../dataset/bdd100k/labels/train.txt', '../dataset/bdd100k/images/train', cls)
    build_labels('../dataset/bdd100k/labels/bdd100k_labels_images_val.json',
                 '../dataset/bdd100k/labels/val.txt', '../dataset/bdd100k/images/val', cls)
