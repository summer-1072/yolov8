import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loss import LossFun
from metric import Metric
from util import time_sync
from tools import load_model
from dataset import LoadDataset
from box import non_max_suppression
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def valid(dataloader, model, hyp, device, training):
    half = hyp['half'] & (device != 'cpu')

    model = model.half() if half else model.float()

    metric = Metric(model.anchor.cls, device)

    loss_mean = None
    loss_fun = LossFun(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['box_w'],
                       hyp['cls_w'], hyp['dfl_w'], model.anchor.reg_max, device)

    cost = 0
    pbar = tqdm(dataloader, file=sys.stderr)

    with torch.no_grad():
        model.eval()
        for index, (imgs, img_infos, labels) in enumerate(pbar):
            t1 = time_sync()
            imgs = imgs.to(device, non_blocking=True)
            imgs = (imgs.half() if half else imgs.float()) / 255
            labels = labels.to(device)

            pred_box, pred_cls, pred_dist, grid, grid_stride = model(imgs)
            preds = torch.cat((pred_box * grid_stride, pred_cls.sigmoid()), 2)
            preds = non_max_suppression(preds, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
                                        hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])

            t2 = time_sync()
            cost += t2 - t1

            loss_items = loss_fun(labels, pred_cls, pred_box, pred_dist, grid, grid_stride)[1]
            loss_mean = (loss_mean * index + loss_items) / (index + 1) if loss_mean is not None else loss_items

            metric.update(labels, preds, img_infos)

            pbar.set_description(metric.desc_head)

        metric.build()

        print(metric.desc_body, file=sys.stderr)

        if training:
            model.float()

        else:
            print(f'speed: ({cost / len(dataloader.dataset):.3})s per image', file=sys.stderr)
            metric.print_details()

        loss_record = [round(x, 4) for x in loss_mean.tolist()]

        return {**dict(zip(['val/' + x for x in loss_fun.names], loss_record)), **metric.metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_img_dir', default='../dataset/bdd100k/images/val')
    parser.add_argument('--val_label_path', default='../dataset/bdd100k/labels/val.txt')
    parser.add_argument('--cls_path', default='../dataset/bdd100k/cls.yaml')

    parser.add_argument('--hyp_path', default='../config/hyp/hyp.yaml')
    parser.add_argument('--model_path', default='../config/model/yolov8x.yaml')
    parser.add_argument('--weight_path', default='../config/weight/yolov8x.pth')
    parser.add_argument('--fused', default=True)

    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load cls
    cls = yaml.safe_load(open(args.cls_path, encoding="utf-8"))

    # load hyp
    hyp = yaml.safe_load(open(args.hyp_path, encoding="utf-8"))

    # model
    model = load_model(args.model_path, cls, args.weight_path, args.fused, hyp['shape'])
    model.to(device)
    model.eval()

    val_dataset = LoadDataset(args.val_img_dir, args.val_label_path, hyp, model.anchor.strides[-1], False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=hyp['batch_size'],
                                num_workers=hyp['njobs'], shuffle=True, collate_fn=LoadDataset.collate_fn)

    valid(val_dataloader, model, hyp, device, False)

# if __name__ == "__main__":
#     # load data
#     labels = torch.tensor([
#         [0, 0, 1.2, 1.2, 7.2, 7.2],
#         [0, 1, 4.8, 4.8, 10.8, 10.8],
#         [1, 0, 1.2, 4.8, 3.6, 7.2],
#         [1, 1, 4.8, 1.2, 10.8, 6],
#         [1, 2, 1.2, 8.4, 10.8, 10.8]])
#
#     pred_box = torch.tensor([
#         [[0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [2.05, 0.05, 3.05, 1.05],
#          [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05], [2.05, 1.05, 3.05, 2.05],
#          [0.05, 2.05, 1.05, 3.05], [1.05, 2.05, 2.05, 3.05], [2.05, 2.05, 3.05, 3.05],
#          [0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05]],
#
#         [[0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [2.05, 0.05, 3.05, 1.05],
#          [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05], [2.05, 1.05, 3.05, 2.05],
#          [0.05, 2.05, 1.05, 3.05], [1.05, 2.05, 2.05, 3.05], [2.05, 2.05, 3.05, 3.05],
#          [0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05]]
#     ])
#
#     pred_cls = torch.tensor([
#         [[0.5, 0.1, 0.1], [0.5, 0.1, 0.1], [0.1, 0.1, 0.1],
#          [0.5, 0.1, 0.1], [0.5, 0.1, 0.1], [0.1, 0.5, 0.1],
#          [0.1, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.5, 0.1],
#          [0.5, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.5, 0.1]],
#
#         [[0.1, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.5, 0.1],
#          [0.5, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1],
#          [0.1, 0.1, 0.5], [0.1, 0.1, 0.5], [0.1, 0.1, 0.5],
#          [0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5], [0.1, 0.1, 0.5]]
#     ])
#
#     pred_dist = torch.tensor([
#         [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]],
#
#         [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
#          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]]
#     ])
#
#     grid = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
#                          [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
#                          [0.5, 2.5], [1.5, 2.5], [2.5, 2.5],
#                          [0.5, 0.5], [1.5, 0.5],
#                          [0.5, 1.5], [1.5, 1.5]
#                          ])
#
#     grid_stride = torch.tensor([[4], [4], [4], [4], [4], [4], [4], [4], [4], [6], [6], [6], [6]])
#
#     img_sizes = torch.tensor([[[12, 12], [12, 12]], [[12, 12], [12, 12]]])
#
#     img_infos = [{'shape': (12, 12), 'ratio': 1, 'offset': (0, 0)},
#                  {'shape': (12, 12), 'ratio': 1, 'offset': (0, 0)},
#                  {'shape': (12, 12), 'ratio': 1, 'offset': (0, 0)},
#                  {'shape': (12, 12), 'ratio': 1, 'offset': (0, 0)}]
#
#     hyp = {'alpha': 1, 'beta': 1, 'topk': 5, 'box_w': 1, 'cls_w': 1, 'dfl_w': 1, 'conf_t': 0.25,
#            'multi_label': False, 'max_box': 30000, 'max_wh': 7680, 'iou_t': 0.7, 'max_det': 300, 'merge': False}
#
#     device = 'cpu'
#
#     loss_fun = LossFun(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['box_w'], hyp['cls_w'], hyp['dfl_w'], 3, device)
#
#     metric = Metric(['car', 'person', 'bike'], device)
#
#     preds = torch.cat((pred_box * grid_stride, pred_cls), 2)
#
#     loss_items = torch.zeros(3, device=device)
#
#     loss_items += loss_fun(labels, pred_cls, pred_box, pred_dist, grid, grid_stride)[1]
#
#     preds = non_max_suppression(preds, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
#                                 hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])
#
#     metric.update(labels, preds, img_infos)
#
#     metric.build()
#
#     print({**dict(zip(['val/box_loss', 'val/cls_loss', 'val/dfl_loss'],
#                       [round(x, 4) for x in (loss_items).tolist()])), **metric.metrics})
#
#     metric.print_details()
