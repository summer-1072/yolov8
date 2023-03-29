import os
import yaml
import json
import math
import torch
import argparse
import numpy as np
from torch import nn
from loss import Loss
from tqdm import tqdm
from valid import valid
from copy import deepcopy
from torch.cuda import amp
from tools import load_model
from dataset import LoadDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class EMA:  # exponential moving average
    def __init__(self, model, decay, tau):
        self.model = deepcopy(model).eval()
        self.updates = 0
        self.decay_fun = lambda x: decay * (1 - math.exp(-x / tau))

    def update(self, model):
        self.updates += 1
        decay = self.decay_fun(self.updates)

        model_dict = self.model.state_dict()
        for k in model_dict.keys():
            if model_dict[k].dtype.is_floating_point:
                model_dict[k] = model_dict[k] * decay + (1 - decay) * model.state_dict()[k].detach()

        self.model.load_state_dict(model_dict)


class EarlyStop:
    def __init__(self, patience):
        self.best_epoch = 0
        self.best_fitness = 0.0
        self.patience = patience or float('inf')

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness

        stop = epoch - self.best_epoch >= self.patience

        if stop:
            print(f'stop training early at {epoch}th epoch, the best one is {self.best_epoch}th epoch')

        return stop


class Train:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        # load cls
        self.cls = yaml.safe_load(open(args.cls_path, encoding="utf-8"))

        # load hyp
        self.hyp = yaml.safe_load(open(args.hyp_path, encoding="utf-8"))

        # model
        self.model = load_model(args.model_path, self.cls, args.fused, args.weight_path, True)
        self.model.to(device)
        self.model.train()

        # optimizer
        self.optimizer = self.build_optimizer(self.hyp['optim'], self.hyp['lr'],
                                              self.hyp['momentum'], self.hyp['weight_decay'])

        # scheduler
        self.lr_fun, self.scheduler = self.build_scheduler(self.hyp['one_cycle'], self.hyp['lrf'], self.hyp['epochs'])

        # loss
        self.loss = Loss(self.hyp['alpha'], self.hyp['beta'], self.hyp['topk'], self.hyp['box_w'],
                         self.hyp['cls_w'], self.hyp['dfl_w'], self.model.anchor.reg_max, device)

        # ema
        self.ema = EMA(self.model, self.hyp['ema_decay'], self.hyp['tau'])

        # early stop
        self.stopper = EarlyStop(self.hyp['patience'])

        # auto mixed precision
        self.amp = device != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.amp)

        # dataset
        train_dataset = LoadDataset(args.train_img_dir, args.train_label_path, self.hyp, True)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.hyp['batch_size'],
                                           num_workers=self.hyp['njobs'], shuffle=True,
                                           collate_fn=LoadDataset.collate_fn)

        val_dataset = LoadDataset(args.val_img_dir, args.val_label_path, self.hyp, False)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.hyp['batch_size'],
                                         num_workers=self.hyp['njobs'], shuffle=True, collate_fn=LoadDataset.collate_fn)

        self.start_epoch = 0
        self.num_batches = len(self.train_dataloader)
        self.accumulate = max(round(self.hyp['num_batch_size'] / self.hyp['batch_size']), 1)
        self.warmup_max = max(round(self.hyp['warmup_epoch'] * self.num_batches), 100)

        # resume train
        if self.args.log_dir:
            self.start_epoch = self.resume_train()

        # build log dir
        else:
            os.makedirs('../log/train', exist_ok=True)
            ord = max([int(x[5:]) for x in os.listdir('../log/train')]) + 1 if len(os.listdir('../log/train')) else 1
            self.args.log_dir = os.path.join('../log/train/train' + str(ord))
            os.makedirs(self.args.log_dir, exist_ok=True)
            os.makedirs(os.path.join(self.args.log_dir, 'weight'), exist_ok=True)

    def build_optimizer(self, optim, lr, momentum, decay):
        params = [[], [], []]
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                params[0].append(v.bias)
            if hasattr(v, 'weight') and isinstance(v, nn.BatchNorm2d):
                params[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                params[2].append(v.weight)

        if optim == 'Adam':
            optimizer = torch.optim.Adam(params[0], lr=lr, betas=(momentum, 0.999))
        elif optim == 'AdamW':
            optimizer = torch.optim.AdamW(params[0], lr=lr, betas=(momentum, 0.999))
        elif optim == 'RMSProp':
            optimizer = torch.optim.RMSprop(params[0], lr=lr, momentum=momentum)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(params[0], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {optim} not implemented.')

        optimizer.add_param_group({'params': params[1], 'weight_decay': 0.0})
        optimizer.add_param_group({'params': params[2], 'weight_decay': decay})

        return optimizer

    def build_scheduler(self, one_cycle, lrf, epochs):
        if one_cycle:
            lr_fun = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
        else:
            lr_fun = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear

        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fun)

        return lr_fun, scheduler

    def save_train(self, epoch):
        param = {'start_epoch': epoch, 'best_epoch': self.stopper.best_epoch,
                 'best_fitness': self.stopper.best_fitness, 'updates': self.ema.updates}

        with open(os.path.join(self.args.log_dir, 'train.json'), 'w') as f:
            json.dump(param, f)

        torch.save(self.model, os.path.join(self.args.log_dir, 'weight', 'model.pth'))
        torch.save(self.ema.model, os.path.join(self.args.log_dir, 'weight', 'ema.pth'))
        torch.save(self.optimizer, os.path.join(self.args.log_dir, 'weight', 'optim.pth'))

    def resume_train(self):
        with open(os.path.join(self.args.log_dir, 'train.json'), 'r') as f:
            param = json.load(f)

        start_epoch = param['start_epoch']
        best_epoch = param['best_epoch']
        best_fitness = param['best_fitness']
        updates = param['updates']

        self.model.load_state_dict(torch.load(os.path.join(self.args.log_dir, 'weight', 'model.pth')))
        self.ema.model.load_state_dict(torch.load(os.path.join(self.args.log_dir, 'weight', 'ema.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.args.log_dir, 'weight', 'optim.pth')))
        self.ema.updates = updates
        self.scheduler.last_epoch = start_epoch - 1
        self.stopper.best_epoch = best_epoch
        self.stopper.best_fitness = best_fitness

        return start_epoch

    def exec_train(self):
        last_step = -1
        for epoch in range(self.start_epoch, self.hyp['epochs']):
            if epoch >= self.hyp['close_mosaic']:
                self.train_dataloader.dataset.hyp['mosaic'] = False

            if epoch >= self.hyp['close_affine']:
                self.train_dataloader.dataset.hyp['affine'] = False

            record_loss = None
            self.optimizer.zero_grad()
            pbar = tqdm(self.train_dataloader, total=self.num_batches, desc="Epoch {}".format(epoch + 1))
            for index, (imgs, img_sizes, labels) in enumerate(pbar):
                # warmup
                count = index + self.num_batches * epoch
                if count <= self.warmup_max:
                    x_in = [0, self.warmup_max]
                    y_in = [1, self.hyp['num_batch_size'] / self.hyp['batch_size']]
                    self.accumulate = max(1, np.interp(count, x_in, y_in).round())

                    for i, param in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr, all other lrs rise from 0.0 to lr
                        y_in = [self.hyp['warmup_bias_lr'] if i == 0 else 0.0, param['initial_lr'] * self.lr_fun(epoch)]
                        param['lr'] = np.interp(count, x_in, y_in)

                        if 'momentum' in param:
                            y_in = [self.hyp['warmup_momentum'], self.hyp['momentum']]
                            param['momentum'] = np.interp(count, x_in, y_in)

                # forward
                with torch.cuda.amp.autocast(self.amp):
                    imgs = imgs.to(self.device, non_blocking=True).float() / 255
                    pred_box, pred_cls, pred_dist, grid, grid_stride = self.model(imgs)

                    loss, loss_items = self.loss(labels, pred_box, pred_cls, pred_dist, grid, grid_stride)

                    record_loss = (record_loss * index + loss_items) / (index + 1) if record_loss is not None  else loss_items

                # backward
                self.scaler.scale(loss).backward()

                if count - last_step >= self.accumulate:
                    # optimizer step
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # ema step
                    self.ema.update(self.model)

                    last_step = count

                # log
                memory = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(
                    ('%12s' * (4 + record_loss.shape[0])) %
                    (f"{epoch + 1}/{self.hyp['epochs']}", memory, *record_loss, labels.shape[0], self.hyp['shape']))

            # scheduler step
            self.scheduler.step()

            # validation
            metric = valid(self.val_dataloader, self.model, self.hyp, self.device, True)

            # early stopping
            stop = self.stopper(epoch, metric['fitness'])

            # save train
            self.save_train(epoch)

            if stop:
                valid(self.val_dataloader, self.model, self.hyp, self.device, False)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', default='../dataset/bdd100k/images/train')
    parser.add_argument('--train_label_path', default='../dataset/bdd100k/labels/train.txt')
    parser.add_argument('--val_img_dir', default='../dataset/bdd100k/images/val')
    parser.add_argument('--val_label_path', default='../dataset/bdd100k/labels/val.txt')
    parser.add_argument('--cls_path', default='../dataset/bdd100k/cls.yaml')

    parser.add_argument('--hyp_path', default='../config/hyp/hyp.yaml')
    parser.add_argument('--model_path', default='../config/model/yolov8x.yaml')
    parser.add_argument('--weight_path', default='../config/weight/yolov8x.pth')
    parser.add_argument('--training', default=True)
    parser.add_argument('--fused', default=True)

    parser.add_argument('--pretrain_dir', default='')
    parser.add_argument('--log_dir', default='')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train = Train(args, device)
    train.exec_train()
