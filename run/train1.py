import os
import yaml
import json
import math
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from torch.cuda import amp
from tools import load_model
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from plot import plot_images, plot_labels
from dataset import build_labels, LoadDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class Optim:
    def __init__(self, model, hyp, num_batch, device):
        self.hyp = hyp
        self.accumulate = max(round(hyp['num_batch_size'] / hyp['batch_size']), 1)
        self.warmup_max = max(round(hyp['warmup_epoch'] * num_batch), 100)

        self.amp = device != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.amp)
        self.optimizer = self.build_optimizer(model, hyp['optim'], hyp['lr'], hyp['momentum'], hyp['decay'])
        self.lr_fun, self.scheduler = self.build_scheduler(self.optimizer, hyp['epochs'], hyp['one_cycle'], hyp['lrf'])

    def build_optimizer(self, model, optim, lr, momentum, decay):
        params = [[], [], []]
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                params[2].append(v.bias)
            if hasattr(v, 'weight') and isinstance(v, nn.BatchNorm2d):
                params[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                params[0].append(v.weight)

        if optim == 'Adam':
            optimizer = torch.optim.Adam(params[2], lr=lr, betas=(momentum, 0.999))
        elif optim == 'AdamW':
            optimizer = torch.optim.AdamW(params[2], lr=lr, betas=(momentum, 0.999))
        elif optim == 'RMSProp':
            optimizer = torch.optim.RMSprop(params[2], lr=lr, momentum=momentum)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(params[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {optim} not implemented.')

        optimizer.add_param_group({'params': params[0], 'weight_decay': decay})
        optimizer.add_param_group({'params': params[1], 'weight_decay': 0.0})

        return optimizer

    def build_scheduler(self, optimizer, epochs, one_cycle, lrf):
        if one_cycle:
            lr_fun = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
        else:
            lr_fun = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fun)

        return lr_fun, scheduler

    def warm_up(self, count, epoch):
        if count < self.warmup_max:
            self.accumulate = max(1, np.interp(count, [0, self.warmup_max],
                                               [1, self.hyp['num_batch_size'] / self.hyp['batch_size']]).round())

            for i, param in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr, weight lr rise from 0.0 to lr
                param['lr'] = np.interp(count, [0, self.warmup_max],
                                        [self.hyp['warmup_bias_lr'] if i == 0 else 0.0,
                                         param['initial_lr'] * self.lr_fun(epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(count, [0, self.warmup_max],
                                                  [self.hyp['warmup_momentum'], self.hyp['momentum']])

    def optimizer_step(self, model):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def scheduler_step(self):
        self.scheduler.step()


class EMA:  # exponential moving average
    def __int__(self, model, decay=0.9999, tau=2000, updates=0):
        self.model = deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.decay_fun = lambda x: decay * (1 - math.exp(-x / tau))
        self.updates = updates

    def update(self, model):
        self.updates += 1
        decay = self.decay_fun(self.updates)

        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v = v * decay + (1 - decay) * model.state_dict()[k].detach()


class EarlyStop:
    def __init__(self, best_epoch=0, best_fitness=0.0, patience=50):
        self.best_epoch = best_epoch
        self.best_fitness = best_fitness
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
    def __int__(self, args, hyp, device):
        self.args = args
        self.hyp = hyp
        self.device = device

    def setup_train(self):
        # model
        self.model = load_model(self.args.model_path, self.args.training, self.args.fused, self.model_weight_path)
        self.model.to(self.device)

        # ema
        self.ema = EMA(self.model, self.hyp['decay'], self.hyp['tau'], self.updates)

        # early stop
        self.stopper, self.stop = EarlyStop(self.best_epoch, self.best_fitness, self.args.patience), False

        # dataset
        train_dataset = LoadDataset(self.args.train_img_dir, self.args.train_label_file, self.hyp)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size,
                                           num_workers=self.args.njobs, shuffle=True, collate_fn=LoadDataset.collate_fn)

    def resume_train(self):
        pass

    def exec_train(self):
        self.setup_train()


parser = argparse.ArgumentParser()
parser.add_argument('--train_img_dir', type=str, default='../dataset/bdd100k/images/train')
parser.add_argument('--train_label_file', type=str, default='../dataset/bdd100k/labels/train.txt')
parser.add_argument('--val_img_dir', type=str, default='../dataset/bdd100k/images/val')
parser.add_argument('--val_label_file', type=str, default='../dataset/bdd100k/labels/val.txt')
parser.add_argument('--cls_file', type=str, default='../dataset/bdd100k/cls.yaml')

parser.add_argument('--hyp_file', type=str, default='../config/hyp/hyp.yaml')
parser.add_argument('--model_file', type=str, default='../config/model/yolov8x.yaml')
parser.add_argument('--weight_file', type=str, default='../config/weight/yolov8x.pth')
parser.add_argument('--training', type=bool, default=True)
parser.add_argument('--fused', type=bool, default=True)

parser.add_argument('--pretrain_dir', type=str, default='')
parser.add_argument('--log_dir', type=str, default='../log/train')
parser.add_argument('--batch_size', type=str, default=2)
parser.add_argument('--njobs', type=str, default=1)
args = parser.parse_args()

if __name__ == "__main__":
    # build label
    if not os.path.exists(args.train_label_file) or not os.path.exists(args.val_label_file):
        print('build yolo labels')
        cls = yaml.safe_load(open('../dataset/bdd100k/cls.yaml', encoding="utf-8"))
        build_labels('../dataset/bdd100k/labels/train.json',
                     args.train_label_file, args.train_img_dir, cls)
        build_labels('../dataset/bdd100k/labels/val.json',
                     args.val_label_file, args.val_img_dir, cls)

    train(args)
