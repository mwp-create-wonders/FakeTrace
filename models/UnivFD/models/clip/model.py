from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .SFSConv import *

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """
    3x3 conv with padding
    in_planes: 输入通道数
    out_planes: 输出通道数
    Kernel_size：核大小
    stride：步长
    padding：保持特征尺寸不变，扩展
    groups：分组卷积的组数
    bias：通常在batchnorm中使用

    效果：经过卷积，得到同样尺寸的特征，但是通道数依据我的卷积核的个数
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 conv
    
    效果：相当于给原始图像逐像素加权，通道数等于我卷积核个数
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=stride, bias=False, padding=0)


# 基础层
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        # 如果norm_layer参数没有特别设计，默认BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        self.conv1 = conv3x3(inplanes, planes, stride)

        # 归一化层
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # 这里stride没有指定意思是：我只使用默认的，无法指定
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # 是否进行下采样，我可以自行设计
        self.downsample = downsample

        # 这里的stride记录的是我指定的
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差补充
        out += identity
        out = self.relu(out)

        return out


# 初始版本
class BottleneckOrigin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # 先降维
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 在低维度计算3*3残差
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 在bottleneck中的创新设计，如果stride=2，则尺寸减半
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 使用1*1的卷积核显著提高输出的通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        # 下采样操作
        # 步长卷积：可能会丢失信息
        # 最大池化：保留最显著的信息
        # 平均池化：保留整体的信息

        # 如果步长大于1，或者输入和输出通道数不匹配，就进行下采样
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # 设置三层
            # 第一层，直接池化，缩小
            # 第二层，放大到期望通道数
            # 第三层，归一化
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        # 这里使用嵌套的方式写
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        # 注意，这是将原始的x进行下采样，然后将结果用于相加
        # 相当于为了构造残差相加，需要进行采样操作去构造相同的形状
        if self.downsample is not None:
            identity = self.downsample(x)

        # resnet
        out += identity
        out = self.relu3(out)
        
        return out


# SFS-Resnet
# 首层和中间层设计
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,First=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 标准Resnet
        width = int(planes * (base_width / 64.)) * groups
        
        # 如果是first层
        self.first = First
        if self.first:
            self.SFSC1 = FirstSFSConvBR(inplanes, width, kernel_size=(1, 1),norm_layer=norm_layer,padding=0)
        else:
            self.SFSC1 = SFSConvBR(inplanes, width, kernel_size=(1,1),norm_layer=norm_layer,padding=0)

        self.SFSC2 = SFSConvBR(width, width, kernel_size=(3,3), stride=stride, groups=groups, norm_layer=norm_layer)

        self.SFSC3 = SFSConvB(width, planes * self.expansion, kernel_size=(1,1), norm_layer=norm_layer,padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 如果是首层，则直接计算，否则先拆分，再计算
        if self.first:
            x_s_res, x_f_res = self.SFSC1(x)
            x_s, x_f = self.SFSC2((x_s_res, x_f_res))
        else:
            x_s_res, x_f_res = x
            x_s, x_f = self.SFSC1((x_s_res,x_f_res))
            x_s, x_f = self.SFSC2((x_s, x_f))

        x_s, x_f = self.SFSC3((x_s, x_f))

        if self.downsample is not None:
            x_s_res, x_f_res = self.downsample((x_s_res,x_f_res))

        # 残差
        x_s += x_s_res
        x_f += x_f_res

        x_s = self.relu(x_s)
        x_f = self.relu(x_f)

        return x_s, x_f


# 尾层设计
class BottleneckLast(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups

        self.SFSC1 = SFSConvBR(inplanes, width,kernel_size=(1,1),padding=0)
        self.SFSC2 = SFSConvBR(width, width, kernel_size=(3, 3), stride=stride, groups=groups, norm_layer=norm_layer)
        self.SFSC3 = LastSFSConvB(width, planes * self.expansion, kernel_size=(1, 1), norm_layer=norm_layer, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):

        x_s_res, x_f_res = x
        x_s, x_f = self.SFSC1((x_s_res, x_f_res))

        x_s, x_f = self.SFSC2((x_s, x_f))

        # 最后只得到一个输出
        x_s = self.SFSC3((x_s, x_f))

        if self.downsample is not None:
            x_s_res = self.downsample((x_s_res, x_f_res))

        # 残差这部分只使用第一个参数的
        x_s += x_s_res
        x_s = self.relu(x_s)

        return x_s


# 多头注意力
class AttentionPool2d(nn.Module):
    """
    AttentionPool2d 实现了一个基于多头注意力机制的池化层，用于将二维特征图的全局信息聚合到一个表示中。
    它通过添加全局平均池化的“伪位置”和位置嵌入，将特征图的每个位置的特征进行聚合。
    最终输出是一个全局表示，可以用于后续的分类或其他任务
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 给每个位置添加位置信息，定义形状
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 定义key,query,value和目标维度
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # 展平
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> NC(HW) -> (HW)NC
        # 添加全局平均池化
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 嵌入位置
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
        

# SFS-Resnet
class ModifiedResNet(nn.Module):
    """
    残差+注意力
    """
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.inplanes = width
        self.groups = 1
        self.base_width = width

        # the 3-layer stem
        # ****************************************原来的三层******************************************************
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        
        # ********************************************************************************************************         
        # residual layers
        # 维度逐渐扩大
        self.layer1 = self._make_layer(width, layers[0], First=True) # 第一层
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2) # stride只会影响每一个Bottleneck的第二层，其他都是默认1
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2) # stride只会影响每一个Bottleneck的第二层，其他都是默认1
        self.layer4 = self._make_last_layer(width * 8, layers[3], stride=2) # 尾层
             
        # 最后添加一个QFV层
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, First=False,norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                SFSConvB(in_channels=self.inplanes,out_channels=planes * Bottleneck.expansion, kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []

        # 先添加模块的第一层，而且只有第一次使用downsample，为了调整维度
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        
        # 确定好输入层大小，直接通过公式计算得到
        self.inplanes = planes * Bottleneck.expansion

        # 中间层，自动依据设计层数添加，而且自动减一
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)
        
        
    def _make_last_layer(self, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        downsample = None
        downsample_last = None
        
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                SFSConvB(in_channels=self.inplanes,out_channels=planes * Bottleneck.expansion, kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []

        # 首先添加一层（带有下采样）
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * Bottleneck.expansion

        # 添加中间层
        for _ in range(1, blocks - 1):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        # 最后添加尾，合二为一
        if self.inplanes != planes * Bottleneck.expansion:
            downsample_last = nn.Sequential(
                LastSFSConvB(in_channels=self.inplanes,out_channels=planes * Bottleneck.expansion, kernel_size=(1,1), stride=stride, padding=0)
            )
        
        layers.append(BottleneckLast(self.inplanes, planes, stride, downsample_last, self.groups,
                            self.base_width, norm_layer))
        
        self.inplanes = planes * Bottleneck.expansion
        
        return nn.Sequential(*layers)
    

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        
        x_s, x_f = self.layer1(x)
        x_s, x_f = self.layer2((x_s,x_f))
        x_s, x_f = self.layer3((x_s,x_f))
        x_s = self.layer4((x_s,x_f))

        # 最后添加一个自注意层
        x = self.attnpool(x_s)

        return x


# 两个函数
class LayerNorm(nn.LayerNorm):
    """解决fp16训练时数值不稳定的问题"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """一种激活函数，sigmod增加参数近似GELU"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# 构建VisionTransformer
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # 添加残差连接
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # 包含多个残差注意力块
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        out = {}
        # 遍历所有层，获取中间层的特征
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature
        return out, x 


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))



    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # 添加CLS和位置信息
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # 特征提取模块
        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        
        out['before_projection'] = x  

        if self.proj is not None:
            # 表示执行矩阵乘法
            x = x @ self.proj
        out['after_projection'] = x 

        # Return both intermediate features and final clip feature 
        # return out
        
        # This only returns CLIP features 
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        # 初始化每一个块
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        # 返回第一个卷积层权重的数据类型
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        
        # 最终得到文本特征表示
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to fp16
    将特定层的参数转化为fp16类型的
    """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    # 得到参数字典，将参数字典用于构建CLIP
    vit = "visual.proj" in state_dict

    # 提取两种网络的配置参数
    # 这里仅仅初始化参数
    # 依据我的选择的参数名称初始化不同的参数
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: # 否则就是ModifiedResNet
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 构建模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    # 加载模型
    model.load_state_dict(state_dict)
    return model.eval()
