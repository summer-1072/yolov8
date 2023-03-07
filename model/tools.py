import yaml
from modules import *
from yolo import YOLO


# 模型参数转移函数
def transform_weight(model, pretrain):
    for m in model.modules():
        if isinstance(m, Conv) and hasattr(m, 'bn'):
            conv = m.conv
            conv.bias = torch.nn.Parameter(torch.zeros(conv.weight.size(0), device=conv.weight.device))
            delattr(m, 'bn')
            m.forward = m.forward_fuse

    model_dict = model.state_dict()
    pret_dict = pretrain.state_dict()
    model_list = [[k, v] for k, v in model_dict.items()]
    pret_list = [[k, v] for k, v in pret_dict.items()]

    assert len(model_list) == len(pret_list)

    for i in range(len(model_list)):
        if model_list[i][1].shape == pret_list[i][1].shape:
            model_list[i][1] = pret_list[i][1]
        else:
            print('layers miss, model layer: %s,  pret layer: %s' % (model_list[i][0], pret_list[i][0]))

    for k, v in model_list:
        model_dict[k] = v

    model.load_state_dict(model_dict)


def load_weight(model, weight_path, training, fused):
    weight_dict = torch.load(weight_path)['model']

    if fused:
        if training:
            model_dict = model.state_dict()
            for k, v in weight_dict.items():
                # conv: weight
                if k in model_dict:
                    model_dict[k] = v
                # bn: bias
                else:
                    model_dict[k.replace('conv.', 'bn.')] = v

            model.load_state_dict(model_dict)

        else:
            # pred -> fuse conv and bn
            for m in model.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    conv = m.conv
                    conv.bias = torch.nn.Parameter(torch.zeros(conv.weight.size(0), device=conv.weight.device))
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse

            model.load_state_dict(weight_dict)

    else:
        model.load_state_dict(weight_dict)


def load_model(model_path, training, fused, weight_path=None):
    args = yaml.safe_load(open(model_path, encoding="utf-8"))
    model = YOLO(args['param'], args['reg_max'], args['chs'], args['strides'], args['num_cls'], training)

    if weight_path:
        load_weight(model, weight_path, training, fused)

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

    w_new = torch.mm(torch.diag(gamma.div(torch.sqrt(eps + var))), w).view(conv.weight.shape)
    b_new = gamma.div(torch.sqrt(eps + var)) * (b - mean) + beta

    m.conv.requires_grad_(False)
    m.conv.weight.copy_(w_new)
    m.conv.bias.copy_(b_new)

    delattr(m, 'bn')
