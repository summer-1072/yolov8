import os
import sys
import yaml
import json
import math
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from valid import valid
from loss import LossFun
from copy import deepcopy
from torch.cuda import amp
from tools import load_model
from dataset import LoadDataset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from plot import plot_labels, plot_images


class EMA:  # exponential moving average
    def __init__(self, model, decay, tau):
        self.model = deepcopy(model).eval()
        self.updates = 0
        self.decay = 0
        self.decay_fun = lambda x: decay * (1 - math.exp(-x / tau))

    def update(self, model):
        self.updates += 1
        self.decay = self.decay_fun(self.updates)

        model_dict = self.model.state_dict()
        for k in model_dict.keys():
            if model_dict[k].dtype.is_floating_point:
                model_dict[k] = model_dict[k] * self.decay + (1 - self.decay) * model.state_dict()[k].detach()

        self.model.load_state_dict(model_dict)


class EarlyStop:
    def __init__(self, patience):
        self.best_epoch = 0
        self.best_fitness = 0.0
        self.patience = patience or float('inf')

    def __call__(self, epoch, fitness):
        best_pth = False
        if fitness >= self.best_fitness:
            best_pth = True
            self.best_epoch = epoch
            self.best_fitness = fitness

        stop = epoch - self.best_epoch >= self.patience

        if stop:
            print(f'stop training early at {epoch}th epoch, the best one is {self.best_epoch}th epoch', file=sys.stderr)

        return stop, best_pth


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


def build_scheduler(optimizer, one_cycle, lrf, epochs):
    if one_cycle:
        lr_fun = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
    else:
        lr_fun = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fun)

    return lr_fun, scheduler


def save_record(epoch, model, ema, optimizer, stopper, best_pth, metrics, log_dir):
    param = {'start_epoch': epoch, 'updates': ema.updates,
             'best_epoch': stopper.best_epoch, 'best_fitness': stopper.best_fitness}

    with open(os.path.join(log_dir, 'param.json'), 'w') as f:
        json.dump(param, f)

    torch.save(model, os.path.join(log_dir, 'weight', 'model.pth'))
    torch.save(ema.model, os.path.join(log_dir, 'weight', 'ema.pth'))
    torch.save(optimizer, os.path.join(log_dir, 'weight', 'optim.pth'))

    if best_pth:
        torch.save(ema.model, os.path.join(log_dir, 'weight', 'best.pth'))

    if os.path.exists(os.path.join(log_dir, 'log.txt')):
        with open(os.path.join(log_dir, 'log.txt'), 'a+') as f:
            f.write(
                (' ' * 6).join([str(epoch + 1).ljust(5)] + [str(v).ljust(len(k)) for k, v in metrics.items()]) + '\n')

    else:
        with open(os.path.join(log_dir, 'log.txt'), 'w+') as f:
            f.write((' ' * 6).join((['epoch'] + list(metrics.keys()))) + '\n')
            f.write(
                (' ' * 6).join([str(epoch + 1).ljust(5)] + [str(v).ljust(len(k)) for k, v in metrics.items()]) + '\n')


def resume_record(model, ema, optimizer, scheduler, stopper, log_dir):
    with open(os.path.join(log_dir, 'param.json'), 'r') as f:
        param = json.load(f)

    start_epoch = param['start_epoch']
    updates = param['updates']
    best_epoch = param['best_epoch']
    best_fitness = param['best_fitness']

    model.load_state_dict(torch.load(os.path.join(log_dir, 'weight', 'model.pth')))
    ema.model.load_state_dict(torch.load(os.path.join(log_dir, 'weight', 'ema.pth')))
    optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'weight', 'optim.pth')))
    ema.updates = updates
    scheduler.last_epoch = start_epoch - 1
    stopper.best_epoch = best_epoch
    stopper.best_fitness = best_fitness

    return start_epoch


def train(args, device):
    # load cls
    cls = yaml.safe_load(open(args.cls_path, encoding="utf-8"))

    # load hyp
    hyp = yaml.safe_load(open(args.hyp_path, encoding="utf-8"))

    # model
    model = load_model(args.model_path, cls, args.weight_path, args.fused)
    model.to(device)

    # ema
    ema = EMA(model, hyp['ema_decay'], hyp['tau'])

    # optimizer
    accumulate = max(round(hyp['total_batch_size'] / hyp['batch_size']), 1)
    weight_decay = hyp['weight_decay'] * hyp['batch_size'] * accumulate / hyp['total_batch_size']
    optimizer = build_optimizer(model, hyp['optim'], hyp['lr'], hyp['momentum'], weight_decay)

    # scheduler
    lr_fun, scheduler = build_scheduler(optimizer, hyp['one_cycle'], hyp['lrf'], hyp['epochs'])

    # loss
    loss_fun = LossFun(hyp['alpha'], hyp['beta'], hyp['topk'], hyp['box_w'], hyp['cls_w'], hyp['dfl_w'],
                       model.anchor.reg_max, device)

    # early stop
    stopper = EarlyStop(hyp['patience'])

    # auto mixed precision
    scaler = amp.GradScaler(enabled=device != 'cpu')

    # dataset
    train_dataset = LoadDataset(args.train_img_dir, args.train_label_path, hyp, model.anchor.strides[-1], True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=hyp['batch_size'],
                                  num_workers=hyp['njobs'], shuffle=True, collate_fn=LoadDataset.collate_fn)

    val_dataset = LoadDataset(args.val_img_dir, args.val_label_path, hyp, model.anchor.strides[-1], False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=hyp['batch_size'],
                                num_workers=hyp['njobs'], shuffle=False, collate_fn=LoadDataset.collate_fn)

    start_epoch = 0
    warmup_max = max(round(hyp['warmup_epoch'] * len(train_dataloader)), 100)

    # resume train
    if args.log_dir:
        start_epoch = resume_record(model, ema, optimizer, scheduler, stopper, args.log_dir)

    # build log_dir
    else:
        os.makedirs('../log/train', exist_ok=True)
        ord = max([int(x[5:]) for x in os.listdir('../log/train')]) + 1 if len(os.listdir('../log/train')) else 1
        args.log_dir = os.path.join('../log/train/train' + str(ord))
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'weight'), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'sample'), exist_ok=True)

    # plot labels
    plot_labels(train_dataset.labels, val_dataset.labels, model.anchor.cls, os.path.join(args.log_dir, 'labels.jpg'))

    # do train
    last_step = -1
    for epoch in range(start_epoch, hyp['epochs']):
        if epoch >= hyp['close_mosaic']:
            hyp['mosaic'] = 0.0

        model.train()

        print('%12s' * (4 + len(loss_fun.names)) % ('epoch', 'memory', *loss_fun.names, 'instances', 'shape'),
              file=sys.stderr)
        loss_mean = None
        optimizer.zero_grad()
        pbar = tqdm(train_dataloader, file=sys.stderr)
        for index, (imgs, img_sizes, labels) in enumerate(pbar):
            # sample plot images
            if index < 5:
                plot_images(imgs, labels, os.path.join(args.log_dir, f'sample/img_{index + 1}.jpg'))

            # warmup
            count = index + len(train_dataloader) * epoch
            if count <= warmup_max:
                x_in = [0, warmup_max]
                y_in = [1, hyp['total_batch_size'] / hyp['batch_size']]
                accumulate = max(1, np.interp(count, x_in, y_in).round())

                for i, param in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr, all other lrs rise from 0.0 to lr
                    y_in = [hyp['warmup_bias_lr'] if i == 0 else 0.0, param['initial_lr'] * lr_fun(epoch)]
                    param['lr'] = np.interp(count, x_in, y_in)

                    if 'momentum' in param:
                        y_in = [hyp['warmup_momentum'], hyp['momentum']]
                        param['momentum'] = np.interp(count, x_in, y_in)

            # forward
            with torch.cuda.amp.autocast(enabled=device != 'cpu'):
                imgs = imgs.to(device, non_blocking=True).float() / 255
                pred_box, pred_cls, pred_dist, grid, grid_stride = model(imgs)

                loss, loss_items = loss_fun(labels, pred_cls, pred_box, pred_dist, grid, grid_stride)
                loss_mean = (loss_mean * index + loss_items) / (index + 1) if loss_mean is not None else loss_items

            # backward
            scaler.scale(loss).backward()

            # optimize
            if count - last_step >= accumulate:
                # optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # ema step
                ema.update(model)

                last_step = count

            # log
            loss_record = [round(x, 4) for x in loss_mean.tolist()]
            pbar.set_description('%12s' * (4 + len(loss_record)) % (
                f"{epoch + 1}/{hyp['epochs']}",
                f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G',
                *loss_record, labels.shape[0], hyp['shape']
            ))

        # scheduler step
        scheduler.step()

        # valid
        metric = valid(val_dataloader, ema.model, hyp, device, True)

        # early stopping
        stop, best_pth = stopper(epoch, metric['metric/fitness'])

        # save train
        loss_record = [round(x, 4) for x in loss_mean.tolist()]
        metrics = {**dict(zip(['train/' + x for x in loss_fun.names], loss_record)), **metric,
                   **{f'lr/pg{i}': round(x['lr'], 4) for i, x in enumerate(optimizer.param_groups)},
                   **{'ema/decay': round(ema.decay, 4)}}

        save_record(epoch, model, ema, optimizer, stopper, best_pth, metrics, args.log_dir)

        if stop:
            valid(val_dataloader, ema.model, hyp, device, False)
            break

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', default='../dataset/bdd10k/images/train')
    parser.add_argument('--train_label_path', default='../dataset/bdd10k/labels/train.txt')
    parser.add_argument('--val_img_dir', default='../dataset/bdd10k/images/val')
    parser.add_argument('--val_label_path', default='../dataset/bdd10k/labels/val.txt')
    parser.add_argument('--cls_path', default='../dataset/bdd10k/cls.yaml')

    parser.add_argument('--hyp_path', default='../config/hyp/hyp.yaml')
    parser.add_argument('--model_path', default='../config/model/yolov8l.yaml')
    parser.add_argument('--weight_path', default='')
    parser.add_argument('--fused', default=False)

    parser.add_argument('--log_dir', default='')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train(args, device)