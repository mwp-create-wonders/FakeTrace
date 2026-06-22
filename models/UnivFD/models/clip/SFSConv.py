import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fractions import Fraction
import math

class MMConv(nn.Module):
    def __init__(self, in_planes):
        super(MMConv, self).__init__()
        self.in_planes = in_planes
        # initialization the weight
        self.const_weight = nn.Parameter(torch.randn(size=[in_planes, 1, 5, 5]), requires_grad=True)
        # 提取所有通道的中心值
        central_pixel = (self.const_weight.data[:, 0, 2, 2])
        
        """归一化每一层的权重，使模型更加关注边缘"""
        for i in range(in_planes):
            # 计算这个卷积核总和
            sumed = self.const_weight.data[i].sum() - central_pixel[i]
            # 归一化参数
            self.const_weight.data[i] /= sumed
            # 将中心参数设置-1.0
            self.const_weight.data[i, 0, 2, 2] = -1.0
        
        # get MMConv kernel
        kernel = torch.FloatTensor(self.const_weight).expand(in_planes, 1, 5, 5)
        # 这两个参数均不可学习
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(in_planes), requires_grad=False)
    
    def forward(self, x):
        #out = F.conv2d(x, self.weight, self.bias, stride=1, padding=2)
        out = F.conv2d(x, self.weight, stride=1, padding=2, groups=self.in_planes)
        return out


# -----basic functions of SFIConv----- #

# 第一层
class FirstSFSConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstSFSConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.s2s = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.f2f = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # frequency modules
        self.MMConv = MMConv(in_channels)
        
    def forward(self, x):
        # 如果步长为2，先均值池化
        if self.stride ==2:
            x = self.AvgPool(x)
        
        # spatial domain
        X_s = self.s2s(x)
        
        # frequency domain
        X_f = self.MMConv(x)   # Multichannel Constrained Separable Conv
        
        X_f = self.f2f(X_f)
        #X_f = F.interpolate(X_f, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)

        return X_s, X_f


# 中间层
class SFSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(SFSConv, self).__init__()
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        #self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha <= 1, "Alpha should be in the interval from 0 to 1."
        self.alpha = alpha
        
        
        self.f2f = None if alpha == 0 else \
                   torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, math.ceil(alpha * groups), bias)
        self.f2s = None if alpha == 0 or alpha == 1 or self.is_dw else \
                   torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.s2f = None if alpha == 1 or alpha == 0 or self.is_dw else \
                   torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.s2s = None if alpha == 1 else \
                   torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, math.ceil(groups - alpha * groups), bias)

    def forward(self, x):
        X_s, X_f = x

        if self.stride ==2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)

        X_s2s = self.s2s(X_s)
        X_f2s = self.f2s(X_f) if not self.is_dw else None
        
        X_f2f = self.f2f(X_f)
        X_s2f = self.s2f(X_s) if not self.is_dw else None
        
        #X_f2s = F.interpolate(X_f2s, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        #X_s2f = F.interpolate(X_s2f, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
        X_s = X_s2s + X_f2s if X_f2s is not None else X_s2s
        X_f = X_s2f + X_f2f if X_s2f is not None else X_f2f

        return X_s, X_f


class LastSFSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastSFSConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.AvgPool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.s2s = torch.nn.Conv2d(in_channels - int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.f2f = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)

        #self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        

    def forward(self, x):
        X_s, X_f = x

        if self.stride ==2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)

        X_s2s = self.s2s(X_s)
        X_f2f = self.f2f(X_f)
        
        #X_f2f = F.interpolate(X_f2f, scale_factor=2, mode='nearest', recompute_scale_factor=True)
        X_out = X_s2s + X_f2f
        
        return X_out


# ---------------SFSConv used in backbone--------------- #
# 带有RELU的首层backbone
class FirstSFSConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstSFSConvBR, self).__init__()
        self.conv = FirstSFSConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))
        
        # 唯一区别
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        x_f = self.relu(self.bn_f(x_f))
        return x_s, x_f


# 不带有RELU的首层backbone
class FirstSFSConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstSFSConvB, self).__init__()
        self.conv = FirstSFSConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.bn_s(x_s)
        x_f = self.bn_f(x_f)
        return x_s, x_f



# 带有RELU的中间层backbone
class SFSConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SFSConvBR, self).__init__()
        self.conv = SFSConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(int(out_channels*(1-alpha)))
        self.bn_f = norm_layer(int(out_channels*alpha))
        
        # 唯一不同
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        x_f = self.relu(self.bn_f(x_f))
        return x_s, x_f


# 不带有RELU的中间层backbone
class SFSConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SFSConvB, self).__init__()
        self.conv = SFSConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_s = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_f = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_s, x_f = self.conv(x)
        x_s = self.bn_s(x_s)
        x_f = self.bn_f(x_f)
        return x_s, x_f


# 带有RELU的尾层backbone
class LastSFSConvBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastSFSConvBR, self).__init__()
        self.conv = LastSFSConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        
        # 唯一不同
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s = self.conv(x)
        x_s = self.relu(self.bn_s(x_s))
        return x_s


# 不带有RELU的尾层backbone
class LastSFSConvB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastSFSConvB, self).__init__()
        self.conv = LastSFSConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_s = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_s = self.conv(x)
        x_s = self.bn_s(x_s)
        return x_s



