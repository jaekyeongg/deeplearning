'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import sys
sys.path.append('..')
from block import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, module=None, image_size=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.image_module = None
        if module == 'SE': self.image_module = SEBlock(self.expansion*planes, 8)
        elif module == 'SEC': self.image_module = SEBlockCon(self.expansion * planes, 8)
        elif module == 'CA': self.image_module = ChannelGate(self.expansion*planes)
        elif module == 'SA': self.image_module = SpatialGate()
        elif module == 'AA': self.image_module = AACN_Layer(self.expansion*planes, image_size=image_size)
        elif module == 'NEW' : self.image_module = NewBlock(self.expansion*planes, 8)
        elif module == 'SE_SA' : self.image_module = nn.Sequential(
            SEBlock(self.expansion*planes, 8),
            SpatialGate()
        )
        elif module == 'SEC_SA' : self.image_module = nn.Sequential(
            SEBlockCon(self.expansion*planes, 8),
            SpatialGate()
        )
        elif module == 'CBAM' : self.image_module = CBAM(self.expansion*planes, 16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        
        if self.image_module is not None:
            out = self.image_module(out)

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, module=None, image_size=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.image_module = None
        if module == 'SE':
            self.image_module = SEBlock(self.expansion * planes, 8)
        elif module == 'SEC':
            self.image_module = SEBlockCon(self.expansion * planes, 8)
        elif module == 'CA':
            self.image_module = ChannelGate(self.expansion * planes)
        elif module == 'SA':
            self.image_module = SpatialGate()
        elif module == 'AA':
            self.image_module = AACN_Layer(self.expansion*planes, image_size=image_size)
        elif module == 'NEW':
            self.image_module = NewBlock(self.expansion * planes, 8)
        elif module == 'SE_SA':
            self.image_module = nn.Sequential(
                SEBlock(self.expansion * planes, 8),
                SpatialGate()
            )
        elif module == 'SEC_SA':
            self.image_module = nn.Sequential(
                SEBlockCon(self.expansion * planes, 8),
                SpatialGate()
            )
        elif module == 'CBAM':
            self.image_module = CBAM(self.expansion * planes, 16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        if self.image_module is not None:
            out = self.image_module(out)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, config, num_classes=100):
        super(ResNet, self).__init__()
        self.config = config
        self.image_module1 = None
        self.image_module2 = None
        self.image_module3 = None
        self.image_module4 = None
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, module=config[0], image_size=32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, module=config[1], image_size=16)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, module=config[2], image_size=8)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, module=config[3], image_size=4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, module, image_size):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, module, image_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.image_module1 is not None :
            out = self.image_module1(out)
        out = self.layer2(out)
        if self.image_module2 is not None :
            out = self.image_module2(out)
        out = self.layer3(out)
        if self.image_module3 is not None :
            out = self.image_module3(out)
        out = self.layer4(out)
        if self.image_module4 is not None :
            out = self.image_module4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(cfg, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], config=cfg, num_classes=num_classes)


def ResNet34(cfg, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], config=cfg, num_classes=num_classes)


def ResNet50(cfg, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], config=cfg, num_classes=num_classes)


def ResNet101(cfg, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], config=cfg, num_classes=num_classes)


def ResNet152(cfg, num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], config=cfg, num_classes=num_classes)


def test():
    net = ResNet18(None, 100)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
