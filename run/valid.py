import torch
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

        self.iou_val = torch.linspace(0.5, 0.95, 10)


    def update_metrics(self, labels, preds, img_sizes):

        for index, pred in enumerate(preds):
            label = labels[labels[:, 0] == index]
            size = img_sizes[index]

            pred[:, :4] = rescale_box(size[0], size[1], pred[:, :4])
            label[:, 2:] = rescale_box(size[0], size[1], label[:, 2:])

            iou = bbox_iou(label[:, 2:].unsqueeze(2), pred[:, :4].unsqueeze(1), 'IoU').squeeze(3).clamp(0)
            cls = label[:, 1:2] == pred[:, 5]


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
