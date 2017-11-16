import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math
import numpy as np


class vgg16(nn.Module):

    def __init__(self, blocks):
        super(vgg16, self).__init__()
        self.blocks = blocks
        self._initialize_weights()

    def forward(self, x):
        features = []
        layer = 0
        for module in self.blocks._modules.values():
            print module
            x = module(x)
            features += [x]
            layer += 1
            if layer == 2:
                break
        return features[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def get_model(pretrained=False, batch_norm=False, **kwargs):
    model = vgg16(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'))
    return model


