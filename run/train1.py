import os
import yaml
import math
import torch
import argparse
from torch import nn
from tqdm import tqdm
from copy import deepcopy
from tools import load_model
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from plot import plot_images, plot_labels
from dataset import build_labels, LoadDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class EMA:  # exponential moving average
    def __int__(self, model, decay=0.9999, tau=2000, updates=0, weight_path=None):
        self.model = deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        if weight_path:
            self.model.load_state_dict(torch.load(weight_path)['model'])

        self.decay_fun = lambda x: decay * (1 - math.exp(-x / tau))
        self.updates = updates

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
        self.device = device

        # model
        model = load_model(self.args.model_path, self.args.training, self.args.fused, self.args.model_weight_path)
        model.to(self.device)
        self.model = model

        # optimizer
        self.optimizer = self.build_optimizer()

        # scheduler
        self.scheduler = self.build_scheduler()

        # ema
        self.ema = EMA(model, hyp['decay'], hyp['tau'], updates, self.args.ema_weight_path)

        # early stop
        self.stopper, self.stop = EarlyStop(self.args.patience), False

        # dataset
        dataset = LoadDataset(args.train_img_dir, args.train_label_file, hyp)
        self.dataloader = DataLoader(dataset=dataset,
                                     batch_size=args.batch_size, num_workers=args.njobs,
                                     shuffle=True, collate_fn=LoadDataset.collate_fn)

    def build_optimizer(self):
        accumulate = max(round(self.args.num_batch_size / self.args.batch_size), 1)
        decay = self.args.decay * self.args.batch_size * accumulate / self.args.num_batch_size
        params = [[], [], []]
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                params[2].append(v.bias)
            if hasattr(v, 'weight') and isinstance(v, nn.BatchNorm2d):
                params[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                params[0].append(v.weight)

        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params[2], lr=self.args.lr, momentum=self.args.momentum, nesterov=True)
        elif self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params[2], lr=self.args.lr, betas=(self.args.momentum, 0.999))
        elif self.args.optimizer == 'RMSProp':
            optimizer = torch.optim.RMSprop(params[2], lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.Adam(params[2], lr=self.args.lr, betas=(self.args.momentum, 0.999))

        optimizer.add_param_group({'params': params[0], 'weight_decay': decay})
        optimizer.add_param_group({'params': params[1], 'weight_decay': 0.0})

        if self.args.optim_weight:
            optimizer.load_state_dict(torch.load(self.args.optim_weight_path)['model'])

        return optimizer

    def build_scheduler(self):
        if self.args.cos_anneal:
            self.lf = lambda x: ((1 - math.cos(x * math.pi / self.args.epochs)) / 2) * (self.args.lrf - 1) + 1
        else:
            self.lf = lambda x: (1 - x / self.args.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear

        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-4)

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # self.scheduler.last_epoch = self.start_epoch - 1
