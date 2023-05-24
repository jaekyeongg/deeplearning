'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

import torch.nn as nn
import torch.nn.functional as F
import torch

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        scale = x * y.expand_as(x)
        res = x + scale
        return res

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class NewBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(NewBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.sg = SpatialGate()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        x = self.sg(x)
        y = self.fc(y).view(b, c, 1, 1)
        scale = x * y.expand_as(x)
        res = x + scale
        return res   

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        print("features : ", features)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        print("classifier : ", self.classifier)


        # Initialize weights
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif 'SE' == v:
            reduction=8
            layers += [SEBlock(in_channels, reduction)]
        elif v == 'SA':
            layers += [SpatialGate()]
        elif v == 'NEW' :
            reduction=8
            layers += [NewBlock(in_channels, reduction)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],

    'SA_123': [64, 64, 'SA', 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'SA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_1': [64, 64, 'SA', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_12': [64, 64, 'SA', 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_23': [64, 64, 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'SA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'NEW_12': [64, 64, 'NEW', 'M', 128, 128, 'NEW', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],

    # Squeeze-and-Excitation Block
    'SE': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE','M',
            512, 512, 512, 512, 'SE','M'],

    'SE_123': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],
    'SE_234': [64, 64, 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE', 'M',
            512, 512, 512, 512, 'M'],
    'SE_456': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE', 'M',
            512, 512, 512, 512, 'SE', 'M'],

    'SE_12': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],
    'SE_45': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'SE', 'M',
            512, 512, 512, 512, 'SE', 'M'],

}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'],batch_norm=True), num_classes)


#def vgg19_bn():
#    """VGG 19-layer model (configuration 'E') with batch normalization"""
#    return VGG(make_layers(cfg['SA_12'], batch_norm=True))
