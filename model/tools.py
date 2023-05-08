import os
import yaml
from yolo import *


def load_model(model_path, cls, weight_path, fused):
    config = yaml.safe_load(open(model_path, encoding="utf-8"))

    model = eval(config['model'])(config['network'], config['reg_max'], config['chs'], config['strides'], cls)

    if weight_path:
        weight = torch.load(weight_path)

        if fused:
            # fuse conv and bn
            for m in model.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    conv = m.conv
                    conv.bias = torch.nn.Parameter(torch.zeros(conv.weight.size(0), device=conv.weight.device))
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse

        model.load_state_dict(weight.state_dict())

    return model


def fuse_conv_bn(m):
    conv, bn = m.conv, m.bn

    w = conv.weight.view(conv.out_channels, -1)
    b = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias

    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    w_new = nn.Parameter(torch.mm(torch.diag(gamma.div(torch.sqrt(eps + var))), w).view(conv.weight.shape))
    b_new = nn.Parameter(gamma.div(torch.sqrt(eps + var)) * (b - mean) + beta)

    m.conv.requires_grad_(False)
    m.conv.weight = w_new
    m.conv.bias = b_new

    delattr(m, 'bn')

if __name__ == "__main__":
    weight_path = '../log/train/train1/weight/best.pth'
    weight = torch.load(weight_path)

    for m in weight.modules():
        if isinstance(m, Conv) and hasattr(m, 'bn'):
            fuse_conv_bn(m)

    torch.save(weight, os.path.join('../config/weight', 'yolov8s.pth'))
