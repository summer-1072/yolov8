import os
import sys
import cv2
import yaml
import torch
import argparse
import numpy as np
from tools import load_model
from collections import Counter
from util import time_sync, color
from box import letterbox, inv_letterbox, non_max_suppression


def annotate(img, pred, cls):
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
            cv2.rectangle(img, pt1, pt2, color(c), 1)
            cv2.putText(img, cls[c] + ':' + str(v), pt3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color(c), 1, cv2.LINE_AA)

    else:
        info = 'zero objects'

    return img, info


def detect(ori_img, hyp, model, half, cls, device):
    t1 = time_sync()
    pre_img, ratio, offset = letterbox(ori_img, hyp['shape'], model.anchor.strides[-1])
    pre_img = np.ascontiguousarray(pre_img.transpose((2, 0, 1))[::-1])  # HWC to CHW, BGR to RGB
    pre_img = torch.from_numpy(pre_img).to(device)
    pre_img = (pre_img.half() if half else pre_img.float()) / 255
    pre_img = pre_img.unsqueeze(0)

    t2 = time_sync()
    pred_box, pred_cls, pred_dist, grid, grid_stride = model(pre_img)
    pred = torch.cat((pred_box * grid_stride, pred_cls.sigmoid()), 2)

    t3 = time_sync()
    pred = non_max_suppression(pred, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
                               hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])

    pred = pred[0]

    if pred.shape[0] > 0:
        pred[:, :4] = inv_letterbox(pred[:, :4], ori_img.shape[:2], ratio, offset)

    t4 = time_sync()
    img, info = annotate(ori_img, pred, cls)

    return img, info, t1, t2, t3, t4


def predict(args, device):
    # load cls
    cls = yaml.safe_load(open(args.cls_path, encoding="utf-8"))

    # load hyp
    hyp = yaml.safe_load(open(args.hyp_path, encoding="utf-8"))

    half = hyp['half'] & (device != 'cpu')

    # load model
    model = load_model(args.model_path, cls, args.weight_path, args.fused, hyp['shape'])
    model = model.half() if half else model.float()
    model.to(device)

    # build log_dir
    if not os.path.exists(args.log_dir) or len(os.listdir(args.log_dir)) == 0:
        log_dir = os.path.join(args.log_dir, 'detect1')
    else:
        ord = max([int(x.replace('detect', '')) for x in os.listdir(args.log_dir)]) + 1
        log_dir = os.path.join(args.log_dir, 'detect' + str(ord))
    os.makedirs(log_dir)

    # predict image
    if args.img_dir:
        files = sorted(os.listdir(args.img_dir))
        d1, d2, d3, num = 0, 0, 0, len(files)
        print('read image dir: %s, %d images' % (args.img_dir, num), file=sys.stderr)

        with torch.no_grad():
            model.eval()
            for i, file in enumerate(files):
                img = cv2.imread(os.path.join(args.img_dir, file))
                img, info, t1, t2, t3, t4 = detect(img, hyp, model, half, cls, device)
                cv2.imwrite(os.path.join(log_dir, file), img)

                d1 += t2 - t1
                d2 += t3 - t2
                d3 += t4 - t3
                print('image %d/%d' % (i + 1, num), file, info, f'({t4 - t1:.3})s', file=sys.stderr)

        sentence = 'speed: %s pre-process, %s model-process, %s post-process, %s per frame, %s fps'
        d1, d2, d3 = (d1 / num), (d2 / num), (d3 / num)
        d4 = d1 + d2 + d3
        print(sentence % (f'{(d1):.3f}s', f'{(d2):.3f}s', f'{(d3):.3f}s', f'{(d4):.3f}s', f'{(1 / d4):.3f}s'),
              file=sys.stderr)

    # predict video
    elif args.video_dir:
        files = sorted(os.listdir(args.video_dir))
        print('read video dir: %s, %d videos' % (args.video_dir, len(files)), file=sys.stderr)

        with torch.no_grad():
            model.eval()
            for file in files:
                cap = cv2.VideoCapture(os.path.join(args.video_dir, file))
                frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FPS))
                h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                d1, d2, d3, num = 0, 0, 0, frames
                print('read video: %s, %d frames' % (file, frames), file=sys.stderr)

                video = cv2.VideoWriter(os.path.join(log_dir, file[:file.index('.')] + '.mp4'),
                                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for i in range(num):
                    _, frame = cap.read(i)
                    img, info, t1, t2, t3, t4 = detect(frame, hyp, model, half, cls, device)
                    video.write(img)

                    d1 += t2 - t1
                    d2 += t3 - t2
                    d3 += t4 - t3
                    print('frame %d/%d' % (i + 1, num), info, f'({t4 - t1:.3})s', file=sys.stderr)

                sentence = 'speed: %s pre-process, %s model-process, %s post-process, %s per frame, %s fps'
                d1, d2, d3 = (d1 / num), (d2 / num), (d3 / num)
                d4 = d1 + d2 + d3
                print(sentence % (f'{(d1):.3f}s', f'{(d2):.3f}s', f'{(d3):.3f}s', f'{(d4):.3f}s', f'{(1 / d4):.3f}s'),
                      file=sys.stderr)


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default='')
parser.add_argument('--video_dir', default='../dataset/bdd100k/videos')
parser.add_argument('--cls_path', default='../dataset/bdd100k/cls.yaml')

parser.add_argument('--model_path', type=str, default='../config/model/yolov8x.yaml')
parser.add_argument('--weight_path', default='../config/weight/yolov8x.pth')
parser.add_argument('--fused', type=bool, default=True)
parser.add_argument('--hyp_path', type=str, default='../config/hyp/hyp.yaml')
parser.add_argument('--log_dir', type=str, default='../log/detect')
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
predict(args, device)
