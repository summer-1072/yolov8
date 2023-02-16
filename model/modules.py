import torch
from torch import nn
from box import dist2bbox, xyxy2xywh


class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride, pad=None):
        super().__init__()

        pad = kernel // 2 if pad is None else pad
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BottleNeck(nn.Module):
    def __init__(self, ch_in, ch_out, kernels=(3, 3), exp=0.5, shortcut=True):
        super().__init__()

        ch_ = int(ch_out * exp)
        self.conv1 = Conv(ch_in, ch_, kernels[0], 1)
        self.conv2 = Conv(ch_, ch_out, kernels[1], 1)
        self.shortcut = shortcut and ch_in == ch_out

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class C2F(nn.Module):
    def __init__(self, ch_in, ch_out, num, shortcut, exp=0.5):
        super().__init__()

        self.ch_ = int(ch_out * exp)
        self.conv1 = Conv(ch_in, 2 * self.ch_, 1, 1)
        self.conv2 = Conv((2 + num) * self.ch_, ch_out, 1, 1)
        self.m = nn.ModuleList(BottleNeck(self.ch_, self.ch_, (3, 3), 1, shortcut) for _ in range(num))

    def forward(self, x):
        x = list(self.conv1(x).split((self.ch_, self.ch_), 1))
        x.extend(m(x[-1]) for m in self.m)

        return self.conv2(torch.cat(x, 1))


class C3(nn.Module):
    def __init__(self, ch_in, ch_out, num=1, exp=0.5, shortcut=True):
        super().__init__()

        ch_ = int(ch_out * exp)
        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = Conv(ch_in, ch_, 1, 1)
        self.conv3 = Conv(2 * ch_, ch_out, 1, 1)
        self.m = nn.Sequential(*(BottleNeck(ch_, ch_, (1, 3), 1, shortcut) for _ in range(num)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class SPPF(nn.Module):
    def __init__(self, ch_in, ch_out, k):
        super().__init__()

        ch_ = ch_in // 2
        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = Conv(ch_ * 4, ch_out, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)
        y4 = self.max_pool(y3)

        return self.conv2(torch.cat((y1, y2, y3, y4), 1))


class Anchor(nn.Module):
    def __init__(self, num_cls, reg_max, strides, training):
        super().__init__()

        self.num_cls = num_cls
        self.reg_max = reg_max
        self.strides = strides
        self.training = training
        self.num_out = reg_max * 4 + num_cls
        self.shape = None
        self.cpoints = torch.empty(0)
        self.cstrides = torch.empty(0)
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(reg_max, dtype=torch.float).view(1, reg_max, 1, 1))

    def make_centers(self, x):
        cpoints, cstrides = [], []
        for i, stride in enumerate(self.strides):
            B, C, H, W = x[i].shape
            dtype, device = x[i].dtype, x[i].device
            grid_y, grid_x = torch.meshgrid(torch.arange(end=H, dtype=dtype, device=device) + 0.5,
                                            torch.arange(end=W, dtype=dtype, device=device) + 0.5)
            cpoints.append(torch.stack((grid_x, grid_y), 2).view(H * W, 2).expand(1, H * W, 2))
            cstrides.append(torch.full((1, H * W, 1), stride, dtype=dtype, device=device))

        return torch.cat(cpoints, 1), torch.cat(cstrides, 1)

    def forward(self, x):
        shape = x[0].shape  # B、C、H、W
        if self.shape != shape:
            self.shape = shape
            self.cpoints, self.cstrides = self.make_centers(x)

        box, cls = torch.cat([xi.view(shape[0], self.num_out, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_cls), 1)

        B, C, A = box.shape
        dists = self.conv(box.view(B, 4, self.reg_max, A).transpose(2, 1).softmax(1)).view(B, 4, A)

        # B、A、C
        dists = dists.permute(0, 2, 1).contiguous()
        cls = cls.permute(0, 2, 1).contiguous()

        dbox = dist2bbox(dists, self.cpoints) * self.cstrides

        if not self.training:
            dbox = xyxy2xywh(dbox)

        return torch.cat((dbox, cls.sigmoid()), 2)
