import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loss import Loss
from metric import Metric
from dataset import LoadDataset
from torch.utils.data import DataLoader
from box import non_max_suppression, rescale_box, bbox_iou


def valid(dataloader, model, hyp, device):
    loss = Loss(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['reg_max'],
                hyp['box_w'], hyp['cls_w'], hyp['dfl_w'], device)

    half = hyp['half'] & (device != 'cpu')

    model = model.half() if half else model.float()
    model.eval()

    metric = Metric(device)
    loss_items = torch.zeros(3, device=device)

    pbar = tqdm(dataloader)
    for index, (imgs, img_sizes, labels) in enumerate(pbar):
        imgs = (imgs.half() if half else imgs.float()) / 255
        imgs = imgs.to(device)

        pred_box, pred_cls, pred_dist, grid, grid_stride = model(imgs)
        preds = torch.cat((pred_box * grid_stride, pred_cls), 2)

        loss_items += loss(labels, pred_box, pred_cls, pred_dist, grid, grid_stride)[1]

        preds = non_max_suppression(preds, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
                                    hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])

        metric.update_status(labels, preds, img_sizes)

    metric.build_metrics()


if __name__ == "__main__":
    # load data
    labels = torch.tensor([
        [0, 0, 1.2, 1.2, 7.2, 7.2],
        [0, 1, 4.8, 4.8, 10.8, 10.8],
        [1, 0, 1.2, 4.8, 3.6, 7.2],
        [1, 1, 4.8, 1.2, 10.8, 6],
        [1, 2, 1.2, 8.4, 10.8, 10.8]])

    pred_box = torch.tensor([
        [[0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [2.05, 0.05, 3.05, 1.05],
         [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05], [2.05, 1.05, 3.05, 2.05],
         [0.05, 2.05, 1.05, 3.05], [1.05, 2.05, 2.05, 3.05], [2.05, 2.05, 3.05, 3.05],
         [0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05]],

        [[0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [2.05, 0.05, 3.05, 1.05],
         [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05], [2.05, 1.05, 3.05, 2.05],
         [0.05, 2.05, 1.05, 3.05], [1.05, 2.05, 2.05, 3.05], [2.05, 2.05, 3.05, 3.05],
         [0.05, 0.05, 1.05, 1.05], [1.05, 0.05, 2.05, 1.05], [0.05, 1.05, 1.05, 2.05], [1.05, 1.05, 2.05, 2.05]]
    ])

    pred_cls = torch.tensor([
        [[0.5, 0.1, 0.1], [0.5, 0.1, 0.1], [0.1, 0.1, 0.1],
         [0.5, 0.1, 0.1], [0.5, 0.2, 0.1], [0.1, 0.5, 0.1],
         [0.1, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.5, 0.1],
         [0.5, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.5, 0.1]],

        [[0.1, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.5, 0.1],
         [0.5, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1],
         [0.1, 0.1, 0.5], [0.1, 0.1, 0.5], [0.1, 0.1, 0.5],
         [0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5], [0.1, 0.1, 0.5]]
    ])

    pred_dist = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]],

        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2],
         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]]
    ])

    grid = torch.tensor([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                         [0.5, 1.5], [1.5, 1.5], [2.5, 1.5],
                         [0.5, 2.5], [1.5, 2.5], [2.5, 2.5],
                         [0.5, 0.5], [1.5, 0.5],
                         [0.5, 1.5], [1.5, 1.5]
                         ])

    grid_stride = torch.tensor([[4], [4], [4], [4], [4], [4], [4], [4], [4], [6], [6], [6], [6]])

    img_sizes = torch.tensor([[[12, 12], [12, 12]], [[12, 12], [12, 12]]])

    hyp = {'alpha': 1, 'beta': 1, 'topk': 3, 'reg_max': 3, 'box_w': 1, 'cls_w': 1, 'dfl_w': 1,
           'conf_t': 0.25, 'multi_label': False, 'max_box': 30000, 'max_wh': 7680,
           'iou_t': 0.7, 'max_det': 300, 'merge': False}

    device = 'cpu'

    loss = Loss(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['reg_max'],
                hyp['box_w'], hyp['cls_w'], hyp['dfl_w'], device)

    metric = Metric(device)

    preds = torch.cat((pred_box * grid_stride, pred_cls), 2)

    loss_items = torch.zeros(3, device=device)

    loss_items += loss(labels, pred_box, pred_cls, pred_dist, grid, grid_stride)[1]

    print(loss_items)

    preds = non_max_suppression(preds, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
                                hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])

    metric.update_status(labels, preds, img_sizes)

    metrics = metric.build_metrics()
