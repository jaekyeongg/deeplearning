'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


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
    
class AACN_Layer(nn.Module):
    def __init__(self, in_channels, k=0.25, v=0.25, kernel_size=3, num_heads=8, image_size=32, inference=False):
        super(AACN_Layer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = math.floor((in_channels*k)/num_heads)*num_heads 
        # Paper: A minimum of 20 dimensions per head for the keys
        if self.dk / num_heads < 20:
            self.dk = num_heads * 20
        self.dv = math.floor((in_channels*v)/num_heads)*num_heads
        
        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dv % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.padding = (self.kernel_size - 1) // 2
        
        self.conv_out = nn.Conv2d(self.in_channels, self.in_channels - self.dv, self.kernel_size, padding=self.padding).cuda()
        self.kqv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1).cuda()
        
        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        self.rel_encoding_w = nn.Parameter(torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
         
    def forward(self, x):
        batch_size, _, height, width = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)
        
        # After splitting, shape is [batch_size, num_heads, height, width, dkh or dvh]
        k = self.split_heads_2d(k, self.num_heads)
        q = self.split_heads_2d(q, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)
        
        # [batch_size, num_heads, height*width, height*width]
        qk = torch.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh).transpose(2, 3))

        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h
        qk += qr_w

        weights = F.softmax(qk, dim=-1)
        
        if self.inference:
            self.weights = nn.Parameter(weights)
        
        attn_out = torch.matmul(weights, flatten_hw(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, self.dv // self.num_heads, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        # Project heads
        attn_out = self.attn_out(attn_out)
        conv_out = self.conv_out(x)
        out = torch.cat((conv_out, attn_out), dim=1)
        
        return out

    # Split channels into multiple heads.
    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs
    
    # Combine heads (inverse of split heads 2d).
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)
    
    # Compute relative logits for both dimensions.
    def relative_logits(self, q):
        _, num_heads, height, width, dkh = q.size()
        rel_logits_w = self.relative_logits_1d(q, self.rel_encoding_w, height, width, num_heads,  [0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.rel_encoding_h, width, height, num_heads,  [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w
    
    # Compute relative logits along one dimenion.
    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)
        # Collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, height, width, 2 * width - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it
        rel_logits = torch.reshape(rel_logits, (-1, height, width, width))
        # Tile for each head
        rel_logits = torch.unsqueeze(rel_logits, dim=1)
        rel_logits = rel_logits.repeat((1, num_heads, 1, 1, 1))
        # Tile height / width times
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))
        # Reshape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        rel_logits = torch.reshape(rel_logits, (-1, num_heads, height * width, height * width))
        return rel_logits

    # Converts tensor from relative to absolute indexing.
    def rel_to_abs(self, x):
        # [batch_size, num_heads*height, L, 2L−1]
        batch_size, num_heads, L, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch_size, num_heads, L, 1)).cuda()
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads, L - 1)).cuda()
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements.
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x
    
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, module = None, image_size=32):
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
        elif module == 'SA': self.image_module = SpatialGate()
        elif module == 'NEW' : self.image_module = NewBlock(self.expansion*planes, 8)
        elif module == 'AA': self.image_module = AACN_Layer(in_channels=planes, image_size=image_size)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        
        if  self.image_module != None :
            out = self.image_module(out)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
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


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
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

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, module = config[0], image_size=32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, module = config[1], image_size=16)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, module = config[2], image_size=8)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, module = config[3], image_size=4)
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
        if self.image_module1 != None :
            out = self.image_module1(out)
        out = self.layer2(out)
        if self.image_module2 != None :
            out = self.image_module2(out)
        out = self.layer3(out)
        if self.image_module3 != None :
            out = self.image_module3(out)
        out = self.layer4(out)
        if self.image_module4 != None :
            out = self.image_module4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def ResNet18(cfg):
    return ResNet(BasicBlock, [2, 2, 2, 2], config=cfg)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
