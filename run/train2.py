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


class EMAModel:  # exponential moving average
    def __int__(self, model, decay, tau, updates, weight_path):
        self.model = deepcopy(model).eval()
        if weight_path:
            self.model.load_state_dict(weight_path)

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
    def __init__(self, best_epoch, best_fitness, patience=50):
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
        # check resume
        if args.log_dir:
            with open(os.path.join(args.log_dir, 'train.json')) as f:
                param = json.load(f)

        model_weight = os.path.join(args.log_dir, 'weight', 'model.pth') if args.log_dir else args.weight_path
        optim_weight = os.path.join(args.log_dir, 'weight', 'optim.pth') if args.log_dir else ''
        ema_weight = os.path.join(args.log_dir, 'weight', 'ema.pth') if args.log_dir else ''
        start_epoch = param['start_epoch'] if args.log_dir else 0
        best_epoch = param['best_epoch'] if args.log_dir else 0
        best_fitness = param['best_fitness'] if args.log_dir else 0.0
        updates = param['updates'] if args.log_dir else 0

        # model
        self.model = load_model(args.model_path, True, args.fused, model_weight)
        self.model.to(device)

        # optimizer
        self.optimizer = self.build_optimizer(hyp['optim'], hyp['lr'], hyp['momentum'], hyp['decay'], optim_weight)

        # scheduler
        self.lr_fun, self.scheduler = self.build_scheduler(hyp['one_cycle'], hyp['epochs'], hyp['lrf'], start_epoch)

        # ema model
        self.ema_model = EMAModel(self.model, hyp['decay'], hyp['tau'], updates, ema_weight)

        # early stop
        self.stopper = EarlyStop(best_epoch, best_fitness, hyp['patience'])

        # auto mixed precision
        self.amp = device != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.amp)

        # dataset
        train_dataset = LoadDataset(self.args.train_img_dir, self.args.train_label_file, self.hyp, True)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size,
                                           num_workers=self.args.njobs, shuffle=True, collate_fn=LoadDataset.collate_fn)

        val_dataset = LoadDataset(self.args.val_img_dir, self.args.val_label_file, self.hyp, False)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size,
                                         num_workers=self.args.njobs, shuffle=True, collate_fn=LoadDataset.collate_fn)

        # other param
        self.epochs = hyp['epochs']
        self.start_epoch = start_epoch
        self.accumulate = max(round(hyp['num_batch_size'] / hyp['batch_size']), 1)
        self.warmup_max = max(round(hyp['warmup_epoch'] * len(self.train_dataloader)), 100)

    def build_optimizer(self, optim, lr, momentum, decay, weight_path):
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

        if weight_path:
            optimizer.load_state_dict(weight_path)

        return optimizer

    def build_scheduler(self, one_cycle, epochs, lrf, start_epoch):
        if one_cycle:
            lr_fun = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
        else:
            lr_fun = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear

        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fun)

        scheduler.last_epoch = start_epoch - 1

        return lr_fun, scheduler

    def exec_train(self):
        for epoch in


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
