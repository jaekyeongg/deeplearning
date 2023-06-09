'''
Modified from https://github.com/pytorch/vision.git
'''
import torch.nn as nn

import sys
sys.path.append('..')
from block import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        # print("features : ", features)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # print("classifier : ", self.classifier)


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
    cnt_pool = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            cnt_pool += 1
        elif 'SE' == v:
            reduction=8
            layers += [SEBlock(in_channels, reduction)]
        elif 'SEC' == v:
            reduction=8
            layers += [SEBlockCon(in_channels, reduction)]
        elif v=='CA':
            layers += [ChannelGate(in_channels)]
        elif v == 'SA':
            layers += [SpatialGate()]
        elif v == 'AA':
            img_size = 32 if cnt_pool==0 else 32//(cnt_pool**2)
            layers += [AACN_Layer(in_channels=in_channels, image_size=img_size)]
        elif v == 'CBAM':
            layers += [CBAM(in_channels,16)]
        elif v == 'NEW':
            reduction=8
            layers += [NewBlock(in_channels, reduction)]
        elif v == 'NAB':
            layers += [NewAttentionBlock(in_channels)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    # Base model
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],

    # Channel Attention Module
    'CA_123': [64, 64, 'CA', 'M', 128, 128, 'CA', 'M', 256, 256, 256, 256, 'CA', 'M', 512, 512, 512, 512, 'M',
               512, 512, 512, 512, 'M'],
    'CA_234': [64, 64, 'M', 128, 128, 'CA', 'M', 256, 256, 256, 256, 'CA', 'M', 512, 512, 512, 512, 'CA', 'M',
               512, 512, 512, 512, 'M'],
    'CA_345': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'CA', 'M', 512, 512, 512, 512, 'CA', 'M',
               512, 512, 512, 512, 'CA', 'M'],

    # Spatial Attention Module
    'SA_1': [64, 64, 'SA', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_12': [64, 64, 'SA', 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_23': [64, 64, 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'SA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_123': [64, 64, 'SA', 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'SA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],

    # Squeeze-and-Excitation Block with residual net
    'SE_123': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'M',
               512, 512, 512, 512, 'M'],
    'SE_234': [64, 64, 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE', 'M',
               512, 512, 512, 512, 'M'],
    'SE_345': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE', 'M',
               512, 512, 512, 512, 'SE', 'M'],

    # Squeeze-and-Excitation Block with 1x1 conv.
    'SEC_12': [64, 64, 'SEC', 'M', 128, 128, 'SEC', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
               512, 512, 512, 512, 'M'],
    'SEC_123': [64, 64, 'SEC', 'M', 128, 128, 'SEC', 'M', 256, 256, 256, 256, 'SEC', 'M', 512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'],
    'SEC_234': [64, 64, 'M', 128, 128, 'SEC', 'M', 256, 256, 256, 256, 'SEC', 'M', 512, 512, 512, 512, 'SEC', 'M',
                512, 512, 512, 512, 'M'],
    'SEC_345': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'SEC', 'M', 512, 512, 512, 512, 'SEC', 'M',
                512, 512, 512, 512, 'SE', 'M'],

    # Attention Augmented Convolutional Network
    'AA_1': [64, 64, 'AA', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'AA_12': [64, 64, 'AA', 'M', 128, 128, 'AA', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'AA_23': [64, 64, 'M', 128, 128, 'AA', 'M', 256, 256, 256, 256, 'AA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'AA_123': [64, 64, 'AA', 'M', 128, 128, 'AA', 'M', 256, 256, 256, 256, 'AA', 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],

    # SE+SA
    'SE_SA_1': [64, 64, 'SE', 'SA', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'],
    'SE_SA_12': [64, 64, 'SE', 'SA', 'M', 128, 128, 'SE', 'SA', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                 512, 512, 512, 512, 'M'],
    'SE_SA_123': [64, 64, 'SE', 'SA', 'M', 128, 128, 'SE', 'SA', 'M', 256, 256, 256, 256, 'SE', 'SA', 'M', 512, 512,
                  512, 512, 'M',
                  512, 512, 512, 512, 'M'],

    # SEC+SA
    'SEC_SA_1': [64, 64, 'SEC', 'SA', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                 512, 512, 512, 512, 'M'],
    'SEC_SA_12': [64, 64, 'SEC', 'SA', 'M', 128, 128, 'SEC', 'SA', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
                  'M',
                  512, 512, 512, 512, 'M'],
    'SEC_SA_123': [64, 64, 'SEC', 'SA', 'M', 128, 128, 'SEC', 'SA', 'M', 256, 256, 256, 256, 'SEC', 'SA', 'M', 512, 512,
                   512, 512, 'M',
                   512, 512, 512, 512, 'M'],

    # CBAM
    'CBAM_1': [64, 64, 'CBAM', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
               512, 512, 512, 512, 'M'],
    'CBAM_12': [64, 64, 'CBAM', 'M', 128, 128, 'CBAM', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'],
    'CBAM_123': [64, 64, 'CBAM', 'M', 128, 128, 'CBAM', 'M', 256, 256, 256, 256, 'CBAM', 'M', 512, 512, 512, 512, 'M',
                 512, 512, 512, 512, 'M'],

    # Our new model
    'NEW_1': [64, 64, 'NEW', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'NEW_12': [64, 64, 'NEW', 'M', 128, 128, 'NEW', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'NEW_123': [64, 64, 'NEW', 'M', 128, 128, 'NEW', 'M', 256, 256, 256, 256, 'NEW', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],

    # Our new model 2
    'NAB_1': [64, 64, 'NAB', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'NAB_12': [64, 64, 'NAB', 'M', 128, 128, 'NAB', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'NAB_123': [64, 64, 'NAB', 'M', 128, 128, 'NAB', 'M', 256, 256, 256, 256, 'NAB', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
}


def vgg11(num_classes, block_type=None):
    """VGG 11-layer model (configuration "VGG11")"""
    return VGG(make_layers(cfg['VGG11']), num_classes)


def vgg11_bn(num_classes, block_type=None):
    """VGG 11-layer model (configuration "VGG11") with batch normalization"""
    return VGG(make_layers(cfg['VGG11'], batch_norm=True), num_classes)


def vgg13(num_classes, block_type=None):
    """VGG 13-layer model (configuration "VGG13")"""
    return VGG(make_layers(cfg['VGG13']), num_classes)


def vgg13_bn(num_classes, block_type=None):
    """VGG 13-layer model (configuration "VGG13") with batch normalization"""
    return VGG(make_layers(cfg['VGG13'], batch_norm=True), num_classes)


def vgg16(num_classes, block_type=None):
    """VGG 16-layer model (configuration "VGG16")"""
    return VGG(make_layers(cfg['VGG16']), num_classes)


def vgg16_bn(num_classes, block_type=None):
    """VGG 16-layer model (configuration "VGG16") with batch normalization"""
    return VGG(make_layers(cfg['VGG16'], batch_norm=True), num_classes)


def vgg19(num_classes, block_type=None):
    """VGG 19-layer model (configuration "VGG19")"""
    config = 'VGG19' if block_type is None else block_type
    return VGG(make_layers(cfg[config]), num_classes)


def vgg19_bn(num_classes, block_type=None):
    """VGG 19-layer model (configuration "VGG19") with batch normalization"""
    config = 'VGG19' if block_type is None else block_type
    return VGG(make_layers(cfg[config], batch_norm=True), num_classes)
