import torch
import numpy as np
from tqdm import tqdm
from loss import Loss
from box import non_max_suppression, rescale_box, bbox_iou


class Valid:
    def __int__(self, dataloader, hyp, half):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.dataloader = dataloader
        self.hyp = hyp
        self.half = half & (device != 'cpu')
        self.device = device

        self.loss = Loss(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['reg_max'],
                         hyp['box_w'], hyp['cls_w'], hyp['dfl_w'], device)

        self.iouv = torch.linspace(0.5, 0.95, 10)

    def update_metrics(self, labels, preds, img_sizes):

        for index, pred in enumerate(preds):
            label = labels[labels[:, 0] == index]
            size = img_sizes[index]

            pred[:, :4] = rescale_box(size[0], size[1], pred[:, :4])
            label[:, 2:] = rescale_box(size[0], size[1], label[:, 2:])

            iou = bbox_iou(label[:, 2:].unsqueeze(2), pred[:, :4].unsqueeze(1), 'IoU').squeeze(3).clamp(0)
            cls = label[:, 1:2] == pred[:, 5]
            matrix = np.zeros((pred.shape[0], self.iouv.shape[0])).astype(bool)

            for i in range(len(self.iouv)):
                y, x = torch.where((iou >= self.iouv[i]) & cls)
                if len(x):
                    matches = torch.cat((torch.stack((y, x), 1), iou[y, x][:, None]), 1).cpu().numpy()
                    if len(x) > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                    matrix[matches[:, 1].astype(int), i] = True

    def __call__(self, model):
        if self.half:
            model.half()

        model.eval()

        desc = ('%22s' + '%11s' * 6) % ('class', 'images', 'instances', 'P', 'R', 'mAP50', 'mAP50-95')
        num_batches = len(self.dataloader)
        pbar = tqdm(self.dataloader, total=num_batches, desc=desc)
        for index, (imgs, labels, img_sizes) in enumerate(pbar):
            imgs = (imgs.half() if self.half else imgs.float())
            imgs = imgs.to(self.device)

            pred_box, pred_cls, pred_dist, grid, grid_stride = model(imgs)

            loss, loss_items = self.loss(labels, pred_box, pred_cls, pred_dist, grid, grid_stride)

            preds = torch.cat((pred_box * grid_stride, pred_cls), 2)
            preds = non_max_suppression(preds, self.hyp['conf_t'], self.hyp['multi_label'], self.hyp['max_box'],
                                        self.hyp['max_wh'], self.hyp['iou_t'], self.hyp['max_det'], self.hyp['merge'])

            self.update_metrics(labels, preds, img_sizes)


if __name__ == "__main__":
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

    preds = torch.cat((pred_box * grid_stride, pred_cls), 2)

    loss = Loss(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['reg_max'], hyp['box_w'], hyp['cls_w'], hyp['dfl_w'], 'cpu')

    loss, loss_items = loss(labels, pred_box, pred_cls, pred_dist, grid, grid_stride)

    preds = torch.cat((pred_box * grid_stride, pred_cls), 2)
    preds = non_max_suppression(preds, hyp['conf_t'], hyp['multi_label'], hyp['max_box'],
                                hyp['max_wh'], hyp['iou_t'], hyp['max_det'], hyp['merge'])

    # update metric
    iouv = torch.linspace(0.05, 0.2, 4)
    stats = []
    for index, pred in enumerate(preds):
        label = labels[labels[:, 0] == index]
        size = img_sizes[index]

        pred[:, :4] = rescale_box(size[0], size[1], pred[:, :4])
        label[:, 2:] = rescale_box(size[0], size[1], label[:, 2:])

        iou = bbox_iou(label[:, 2:].unsqueeze(1), pred[:, :4].unsqueeze(0), 'IoU').squeeze(2).clamp(0)
        cls = label[:, 1:2] == pred[:, 5]
        matrix = np.zeros((pred.shape[0], iouv.shape[0])).astype(bool)

        for i in range(len(iouv)):
            y, x = torch.where((iou >= iouv[i]) & cls)
            if len(x):
                matches = torch.cat((torch.stack((y, x), 1), iou[y, x][:, None]), 1).cpu().numpy()
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                matrix[matches[:, 1].astype(int), i] = True

        stats.append((torch.tensor(matrix), pred[:, 4], pred[:, 5], label[:, 1]))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    matrix, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]

    index = np.argsort(-conf)
    matrix, conf, pred_cls = matrix[index], conf[index], pred_cls[index]

    cls, count = np.unique(target_cls, return_counts=True)
    num_cls = cls.shape[0]
    P, R, AP = np.zeros((num_cls, 1000)), np.zeros((num_cls, 1000)), np.zeros((num_cls, matrix.shape[1]))

    x = np.linspace(0, 1, 10)
    for index, val in enumerate(cls):
        matches = pred_cls == val

        recall = matrix[matches].cumsum(0) / count[index]

        ans = np.interp(-x, -conf[matches], recall[:, 0], left=0)  # negative x, xp because xp decreases

        print('tp', matrix[matches].cumsum(0))
        print('total', count[index])
        print('recall', recall)
        print('-x', -x)
        print('conf', -conf[matches])

        print('ans', ans)

        assert 0 == 1
