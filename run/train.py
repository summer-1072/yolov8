import os
import yaml
import torch
import argparse
from tqdm import tqdm
from tools import load_model
from torch.utils.data import DataLoader
from plot import plot_images, plot_labels
from dataset import build_labels, LoadDataset


def train(args):
    # load cls
    cls = yaml.safe_load(open(args.cls_file, encoding="utf-8"))

    # load hyp
    hyp = yaml.safe_load(open(args.hyp_file, encoding="utf-8"))

    # select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load dataset
    dataset = LoadDataset(args.train_img_dir, args.train_label_file, hyp)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size, num_workers=args.njobs,
                            shuffle=True, collate_fn=LoadDataset.collate_fn)

    # build log_dir
    if args.pretrain_dir == '':
        if not os.path.exists(args.log_dir) or len(os.listdir(args.log_dir)) == 0:
            log_dir = os.path.join(args.log_dir, 'train1')
        else:
            ord = max([int(x.replace('train', '')) for x in os.listdir(args.log_dir)]) + 1
            log_dir = os.path.join(args.log_dir, 'train' + str(ord))

        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'imgs'))
        weight_file = args.weight_file

    else:
        log_dir = args.pretrain_dir
        weight_file = os.path.join(log_dir, 'weight.pth')

    # plot label
    # plot_labels(dataset.labels, cls, os.path.join(log_dir, 'label.jpg'))

    # load model
    model = load_model(args.model_file, weight_file, args.training, args.fused)
    model.to(device)
    model.train()

    from torch import nn

    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    print('bn', bn)
    for v in model.modules():
        if isinstance(v, nn.BatchNorm2d):
            print(v.weight)
    # pbar = tqdm(dataloader, ncols=100, desc="Epoch {}".format(1))
    # for index, (imgs, labels) in enumerate(pbar):
    #     plot_images(imgs, labels, os.path.join(log_dir + '/imgs', str(index) + '.jpg'))


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

    # train data
    train(args)
