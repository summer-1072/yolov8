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


def build_optimizer(model, optim, lr, momentum, decay):
    params = [[], [], []]
    for v in model.modules():
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


def build_scheduler(optimizer, epochs, one_cycle, lrf):
    if one_cycle:
        lr_fun = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
    else:
        lr_fun = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fun)

    return lr_fun, scheduler


class EMA:  # exponential moving average
    def __int__(self, model, decay=0.9999, tau=2000):
        self.updates = 0
        self.model = deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.decay_fun = lambda x: decay * (1 - math.exp(-x / tau))

    def update(self, model):
        self.updates += 1
        decay = self.decay_fun(self.updates)

        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v = v * decay + (1 - decay) * model.state_dict()[k].detach()


class EarlyStop:
    def __init__(self, patience=50):
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
    def __int__(self, args, hyp, device):
        self.args = args
        self.hyp = hyp
        self.device = device

    def setup_train(self):
        # model
        self.model = load_model(self.args.model_path, self.args.training, self.args.fused, self.args.weight_path)
        self.model.to(self.device)

        # optimizer
        self.optimizer = build_optimizer(self.model, self.hyp['optim'], self.hyp['lr'], self.hyp['momentum'],
                                         self.hyp['decay'])

        # scheduler
        self.lr_fun, self.scheduler = build_scheduler(self.optimizer, self.hyp['epochs'], self.hyp['one_cycle'],
                                                      self.hyp['lrf'])

        # ema
        self.ema = EMA(self.model, self.hyp['decay'], self.hyp['tau'])

        # early stop
        self.stopper, self.stop = EarlyStop(self.hyp['patience']), False

        # dataset
        train_dataset = LoadDataset(self.args.train_img_dir, self.args.train_label_file, self.hyp, True)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size,
                                           num_workers=self.args.njobs, shuffle=True, collate_fn=LoadDataset.collate_fn)

        val_dataset = LoadDataset(self.args.val_img_dir, self.args.val_label_file, self.hyp, False)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size,
                                         num_workers=self.args.njobs, shuffle=True, collate_fn=LoadDataset.collate_fn)

    def resume_train(self, log_dir):
        pass

    def exec_train(self):
        pass


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
