import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fractions import Fraction
import math

# -----MMConv卷积核----- #
class MMConv(nn.Module):
    def __init__(self, in_planes):
        super(MMConv, self).__init__()
        self.in_planes = in_planes
        
        # 初始化随机权重
        self.const_weight = nn.Parameter(torch.randn(size=[1, 1, 5, 5]), requires_grad=True)
        
        # 创建十字掩码
        cross_mask = torch.zeros(5, 5, dtype=torch.bool)
        # 设置十字位置（中心行和中心列）
        cross_mask[2, :] = True  # 中心行
        cross_mask[:, 2] = True  # 中心列
        
        # 分离十字区域和周围区域
        cross_region = self.const_weight.data * cross_mask.float()
        surround_region = self.const_weight.data * (~cross_mask).float()
        
        # 归一化十字区域权重和为+1
        cross_sum = cross_region.sum()
        if cross_sum != 0:
            cross_region = cross_region / cross_sum
        
        # 归一化周围区域权重和为-1
        surround_sum = surround_region.sum()
        if surround_sum != 0:
            surround_region = surround_region / surround_sum * (-1)
        
        # 合并权重
        self.const_weight.data = cross_region + surround_region
        
        # 扩展到所有通道
        kernel = self.const_weight.data.expand(in_planes, 1, 5, 5)
        
        # 固定参数（不可学习）
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(in_planes), requires_grad=False)
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=1, padding=2, groups=self.in_planes)


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



