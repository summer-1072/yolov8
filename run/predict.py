import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from tools import load_model
from util import time_sync, color
from collections import Counter
from box import letterbox, rescale_box, non_max_suppression


def save_results(img, pred, cls, file):
    if len(pred) > 0:
        objs = [[k, v] for k, v in dict(Counter(pred[:, 5].tolist())).items()]
        objs.sort()
        info = ', '.join([str(x[1]) + ' ' + cls[int(x[0])] for x in objs]) + ', Done.'

        for line in pred:
            line = line.tolist()
            pt1 = (round(line[0]), round(line[1]))
            pt2 = (round(line[2]), round(line[3]))
            pt3 = (round(line[0]), round(line[1]) - 8)

            c = int(line[5])
            v = round(line[4], 2)
            cv2.rectangle(img, pt1, pt2, color(c), 4)
            cv2.putText(img, cls[c] + ' ' + str(v), pt3, 0, 0.75, (255, 255, 255), 2)

    else:
        info = 'zero objects'

    cv2.imwrite(file, img)

    return info


def detect(args, device):
    # load hyp
    hyp = yaml.safe_load(open(args.hyp_file, encoding="utf-8"))

    half = hyp['half'] & (device != 'cpu')

    # load model
    model = load_model(args.model_file, args.fused, args.weight_file, False)
    model = model.half() if half else model.float()
    model.to(device)
    model.eval()

    # build log_dir
    if not os.path.exists(args.log_dir) or len(os.listdir(args.log_dir)) == 0:
        log_dir = os.path.join(args.log_dir, 'detect1')
    else:
        ord = max([int(x.replace('detect', '')) for x in os.listdir(args.log_dir)]) + 1
        log_dir = os.path.join(args.log_dir, 'detect' + str(ord))
    os.makedirs(log_dir)

    files = [x for x in os.listdir(args.img_dir)]
    cls = yaml.safe_load(open(args.cls_file, encoding="utf-8"))
    d1, d2, d3, num = 0, 0, 0, len(files)
    for i, file in enumerate(files):
        img0 = cv2.imread(os.path.join(args.img_dir, file))

        t1 = time_sync()
        img1, _, _ = letterbox(img0, hyp['shape'], hyp['stride'])
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = img1 / 255
        img1 = torch.from_numpy(img1)
        img1 = img1.half() if half else img1.float()
        img1 = img1.unsqueeze(0)
        img1 = img1.to(device)

        t2 = time_sync()
        pred = model(img1)

        t3 = time_sync()
        pred = non_max_suppression(pred, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
                                   hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])

        pred = pred[0]

        if pred.shape[0] > 0:
            pred[:, :4] = rescale_box(img0.shape[:2], img1.shape[2:], pred[:, :4])

        t4 = time_sync()
        info = save_results(img0, pred, cls, os.path.join(log_dir, file))

        d1 += t2 - t1
        d2 += t3 - t2
        d3 += t4 - t3
        print('image %d/%d' % (i + 1, num), file, info, f'({t4 - t1:.3})s')

    sentence = 'speed: %s pre-process, %s model-process, %s post-process, %s per image'
    d1 = (d1 / num)
    d2 = (d2 / num)
    d3 = (d3 / num)
    d4 = d1 + d2 + d3
    print(sentence % (f'{(d1):.3f}s', f'{(d2):.3f}s', f'{(d3):.3f}s', f'{(d4):.3f}s'))


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='../dataset/coco/images')
parser.add_argument('--cls_file', type=str, default='../dataset/coco/cls.yaml')
parser.add_argument('--model_file', type=str, default='../config/model/yolov8x.yaml')
parser.add_argument('--weight_file', type=str, default='../config/weight/yolov8x.pth')
parser.add_argument('--fused', type=bool, default=True)
parser.add_argument('--hyp_file', type=str, default='../config/hyp/hyp.yaml')
parser.add_argument('--log_dir', type=str, default='../log/detect')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detect(args, device)
