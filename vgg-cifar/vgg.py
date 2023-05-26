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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

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
    cnt_pool = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            cnt_pool += 1
        elif 'SE' == v:
            reduction=8
            layers += [SEBlock(in_channels, reduction)]
        elif 'AA' == v:
            if cnt_pool==0:
                img_size = 32
            else:
                img_size = 32//(2**cnt_pool)
            layers += [AACN_Layer(in_channels=in_channels, image_size=img_size)]
        elif v == 'SA':
            layers += [SpatialGate()]
        elif v == 'CBAM':
            layers += [CBAM(in_channels,16)]
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

    # Spatial Attention Module
    'SA_123': [64, 64, 'SA', 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'SA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_1': [64, 64, 'SA', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_12': [64, 64, 'SA', 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'SA_23': [64, 64, 'M', 128, 128, 'SA', 'M', 256, 256, 256, 256, 'SA', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],

    # Our new model
    'NEW_12': [64, 64, 'NEW', 'M', 128, 128, 'NEW', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],
    'NEW_123': [64, 64, 'NEW', 'M', 128, 128, 'NEW', 'M', 256, 256, 256, 256, 'NEW', 'M', 512, 512, 512, 512, 'M',
             512, 512, 512, 512, 'M'],

    # Squeeze-and-Excitation Block
    'SE': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE','M',
            512, 512, 512, 512, 'SE','M'],

    'SE_123': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],
    'SE_234': [64, 64, 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE', 'M',
            512, 512, 512, 512, 'M'],
    'SE_345': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'SE', 'M', 512, 512, 512, 512, 'SE', 'M',
            512, 512, 512, 512, 'SE', 'M'],

    'SE_12': [64, 64, 'SE', 'M', 128, 128, 'SE', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],
    'SE_45': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'SE', 'M',
            512, 512, 512, 512, 'SE', 'M'],

    # Attention Augmented Convolutional Network
    'AA_123': [64, 64, 'AA', 'M', 128, 128, 'AA', 'M', 256, 256, 256, 256, 'AA', 'M', 512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'],
    'AA_234': [64, 64, 'M', 128, 128, 'AA', 'M', 256, 256, 256, 256, 'AA', 'M', 512, 512, 512, 512, 'AA', 'M',
            512, 512, 512, 512, 'M'],
    'AA_345': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'AA', 'M', 512, 512, 512, 512, 'AA', 'M',
            512, 512, 512, 512, 'SE', 'M'],

    # SE+SA
    'SE_SA_12': [64, 64, 'SE', 'SA', 'M', 128, 128, 'SE', 'SA','M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
        512, 512, 512, 512, 'M'],
    'SE_SA_123': [64, 64, 'SE', 'SA', 'M', 128, 128, 'SE', 'SA','M', 256, 256, 256, 256, 'SE', 'SA', 'M', 512, 512, 512, 512, 'M',
        512, 512, 512, 512, 'M'],
    
    #CBAM
    'CBAM_12': [64, 64, 'CBAM', 'M', 128, 128, 'CBAM', 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
        512, 512, 512, 512, 'M'],
    'CBAM_123': [64, 64, 'CBAM', 'M', 128, 128, 'CBAM', 'M', 256, 256, 256, 256, 'CBAM', 'M', 512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'],

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


def vgg19(num_classes, block_type):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg[block_type],batch_norm=True), num_classes)


#def vgg19_bn():
#    """VGG 19-layer model (configuration 'E') with batch normalization"""
#    return VGG(make_layers(cfg['SA_12'], batch_norm=True))
