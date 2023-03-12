import torch
from tqdm import tqdm
from loss import Loss
from copy import deepcopy


class Validator:
    def __int__(self, dataloader, training, half, hyp):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.dataloader = dataloader
        self.training = training
        self.half = half & (device != 'cpu')
        self.device = device

        self.loss = Loss(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['reg_max'],
                         hyp['box_w'], hyp['cls_w'], hyp['dfl_w'], device)

    def __call__(self, model):
        if self.half:
            model.half()

        model.eval()

        desc = ('%22s' + '%11s' * 6) % ('class', 'images', 'instances', 'P', 'R', 'mAP50', 'mAP50-95')
        num_batches = len(self.dataloader)
        pbar = tqdm(self.dataloader, total=num_batches, desc=desc)
        for index, imgs in enumerate(pbar):
            pass
