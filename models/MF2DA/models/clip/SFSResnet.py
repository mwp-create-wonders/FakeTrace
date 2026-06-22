from collections import OrderedDict
from typing import Dict, Optional, List, Tuple, Any
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
                 base_width=64, norm_layer=None, alpha=0.5):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups

        self.SFSC1 = SFSConvBR(inplanes, width, kernel_size=(1,1), padding=0, alpha=alpha)
        self.SFSC2 = SFSConvBR(width, width, kernel_size=(3, 3), stride=stride, 
                              groups=groups, norm_layer=norm_layer, alpha=alpha)
        self.SFSC3 = LastSFSConvB(width, planes * self.expansion, 
                                 kernel_size=(1, 1), norm_layer=norm_layer, 
                                 padding=0, alpha=alpha)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.alpha = alpha

    def forward(self, x):
        # print(f"Bottleneck输入: x_s={x[0].shape}, x_f={x[1].shape}")
        
        x_s_res, x_f_res = x
        
        x_s, x_f = self.SFSC1((x_s_res, x_f_res))
        # print(f"conv1后: x_s={x_s.shape}, x_f={x_f.shape}")

        x_s, x_f = self.SFSC2((x_s, x_f))
        # print(f"conv2后: x_s={x_s.shape}, x_f={x_f.shape}")

        x_s = self.SFSC3((x_s, x_f))
        # print(f"conv3后: x_s={x_s.shape}")

        if self.downsample is not None:
            # print("进行的last的下采样")
            x_s_res = self.downsample((x_s_res, x_f_res))
            # print(f"下采样后: x_s_res={x_s_res.shape}")

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
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        # 展平特征图: NCHW -> (HW)NC
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # 形状: (HW, N, C)
        
        # 添加全局平均池化token
        global_token = x.mean(dim=0, keepdim=True)  # 形状: (1, N, C)
        x = torch.cat([global_token, x], dim=0)  # 形状: (HW+1, N, C)
        
        # 动态调整位置编码维度
        current_seq_len = x.shape[0]
        if self.positional_embedding.shape[0] != current_seq_len:
            # 使用插值调整位置编码维度
            pos_embed = F.interpolate(
                self.positional_embedding.unsqueeze(0).permute(0, 2, 1),  # (1, C, L_old)
                size=current_seq_len,
                mode='linear'
            ).permute(0, 2, 1).squeeze(0)  # (L_new, C)
        else:
            pos_embed = self.positional_embedding
        
        # 添加位置编码
        x = x + pos_embed[:, None, :].to(x.dtype)  # 广播到 (HW+1, N, C)
        
        # 注意力机制
        x, _ = F.multi_head_attention_forward(
            query=x[:1],  # 只对全局token做查询
            key=x, 
            value=x,
            embed_dim_to_check=self.embed_dim,
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
        
        return x.squeeze(0)  # 输出形状: (N, output_dim)
        

class LastDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, alpha=0.5, norm_layer=None):
        super(LastDownsample, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.stride = stride
        self.alpha = alpha
        
        # 如果需要空间下采样
        if self.stride == 2:
            self.AvgPool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        
        # 分别对主流和快速流进行卷积
        self.s2s = nn.Conv2d(in_channels - int(alpha * in_channels), out_channels,
                            kernel_size=1, stride=1, bias=False)
        self.f2f = nn.Conv2d(int(alpha * in_channels), out_channels,
                            kernel_size=1, stride=1, bias=False)
        
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        X_s, X_f = x

        # 空间下采样
        if self.stride == 2:
            X_s, X_f = self.AvgPool(X_s), self.AvgPool(X_f)

        # 分别卷积
        X_s2s = self.s2s(X_s)
        X_f2f = self.f2f(X_f)
        
        # 相加合并
        X_out = X_s2s + X_f2f
        X_out = self.bn(X_out)
        
        return X_out


# SFS-Resnet
class ModifiedResNet(nn.Module):
    """
    ModifiedResNet as a standalone feature extractor
    """
    def __init__(self, 
                 layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
                 output_dim: int = 768,
                 heads: int = 8,
                 input_resolution: int = 224,
                 width: int = 64,
                 use_attention_pool: bool = True):
        """
        Args:
            layers: 各阶段的残差块数量，如(3,4,6,3)对应ResNet50
            output_dim: 输出特征的维度
            heads: 注意力头数
            input_resolution: 输入图像分辨率
            width: 基础通道数
            use_attention_pool: 是否使用注意力池化
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.inplanes = width
        self.groups = 1
        self.base_width = width
        self.use_attention_pool = use_attention_pool

        # 3-layer stem
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
        
        # residual layers
        self.layer1 = self._make_layer(width, layers[0], First=True)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_last_layer(width * 8, layers[3], stride=2)
        
        # attention pool or global average pool
        embed_dim = width * 32
        if use_attention_pool:
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            # 如果输出维度不等于嵌入维度，添加线性变换
            self.fc = nn.Linear(embed_dim, output_dim) if output_dim != embed_dim else nn.Identity()

        # 初始化权重
        self._initialize_weights()

        # 添加多层特征融合相关层
        self._init_fusion_projections(768)

    
    def _make_layer(self, planes, blocks, stride=1, First=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                SFSConvB(in_channels=self.inplanes, out_channels=planes * Bottleneck.expansion, 
                         kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.inplanes = planes * Bottleneck.expansion

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
                SFSConvB(in_channels=self.inplanes, out_channels=planes * Bottleneck.expansion, 
                         kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * Bottleneck.expansion

        for _ in range(1, blocks - 1):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        # 在创建BottleneckLast时
        if stride != 1 or self.inplanes != planes * BottleneckLast.expansion:
            # 为最后一层创建专用的单流下采样
            downsample_last = LastDownsample(self.inplanes, planes * BottleneckLast.expansion, 
                                       stride=stride, alpha=0.5, norm_layer=norm_layer)
        else:
            downsample_last = None
        
        layers.append(BottleneckLast(self.inplanes, planes, stride, downsample_last, self.groups,
                            self.base_width, norm_layer))
        
        return nn.Sequential(*layers)
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _init_fusion_projections(self, fusion_dim: int = 768):
        """初始化特征融合的投影层 - 动态版本"""
        # 我们将在forward中动态创建投影层，因为维度可能变化
        self.fusion_dim = fusion_dim
        self.projection_layers_initialized = False
    
        # 注意力机制
        self.layer_attention = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 4, 1)
        )
        
        # 融合处理器
        self.fusion_processor = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),  # 扩展维度
            nn.BatchNorm1d(fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),  # 压缩到目标维度
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True),
        )


    def _init_dynamic_projections(self, actual_dims):
        """根据实际维度动态初始化投影层"""
        # Stem投影
        self.stem_proj = nn.Sequential(
            nn.Linear(actual_dims['stem'], self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer1投影
        self.layer1_proj = nn.Sequential(
            nn.Linear(actual_dims['layer1'], self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer2投影
        self.layer2_proj = nn.Sequential(
            nn.Linear(actual_dims['layer2'], self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer3投影
        self.layer3_proj = nn.Sequential(
            nn.Linear(actual_dims['layer3'], self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer4投影
        self.layer4_proj = nn.Sequential(
            nn.Linear(actual_dims['layer4'], self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        self.projection_layers_initialized = True

        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回所有层的特征 - 修正版
        """
        features = {}
        
        # 原有的主干网络前向传播保持不变
        # Stem features
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x_stem = self.avgpool(x)
        features['stem'] = x_stem
        # print(f"stem后形状 - x_stem: {x_stem.shape}")
        
        # Layer 1 features (dual-stream)
        x_s, x_f = self.layer1(x_stem)
        features['layer1_spatial'] = x_s
        features['layer1_frequency'] = x_f
        # print(f"layer1后形状 - x_s: {x_s.shape}, x_f: {x_f.shape}")
        
        # Layer 2 features (dual-stream)
        x_s, x_f = self.layer2((x_s, x_f))
        features['layer2_spatial'] = x_s
        features['layer2_frequency'] = x_f
        # print(f"layer2后形状 - x_s: {x_s.shape}, x_f: {x_f.shape}")
        
        # Layer 3 features (dual-stream)
        x_s, x_f = self.layer3((x_s, x_f))
        features['layer3_spatial'] = x_s
        features['layer3_frequency'] = x_f
        # print(f"layer3后形状 - x_s: {x_s.shape}, x_f: {x_f.shape}")
        
        # Layer 4 features (single-stream)
        x_final = self.layer4((x_s, x_f))
        features['layer4'] = x_final
        # print(f"layer4后形状 - x_final: {x_final.shape}")
    
        if self.use_attention_pool:
            final_pooled = self.attnpool(x_final)
        else:
            pooled = self.global_pool(x_final)
            pooled = torch.flatten(pooled, 1)
            final_pooled = self.fc(pooled)
    
        features['final_pooled'] = final_pooled
        # print(f"attnpool后的形状 - final_pooled: {final_pooled.shape}")
        
        # ==================== 多层特征融合 - 动态维度 ====================
        # 对每一层特征进行全局平均池化
        
        # 1. Stem特征处理 [B, 64, H, W] -> [B, 64]
        stem_feat = F.adaptive_avg_pool2d(x_stem, (1, 1)).view(x_stem.size(0), -1)  # [B, 64]
        
        # 2. Layer1特征处理 - 空间和频率分支 [B, C, H/4, W/4] -> [B, C]
        layer1_s_pool = F.adaptive_avg_pool2d(x_s, (1, 1)).view(x_s.size(0), -1)
        layer1_f_pool = F.adaptive_avg_pool2d(x_f, (1, 1)).view(x_f.size(0), -1)
        
        # 3. Layer2特征处理 [B, C, H/8, W/8] -> [B, C]
        layer2_s_pool = F.adaptive_avg_pool2d(x_s, (1, 1)).view(x_s.size(0), -1)
        layer2_f_pool = F.adaptive_avg_pool2d(x_f, (1, 1)).view(x_f.size(0), -1)
        
        # 4. Layer3特征处理 [B, C, H/16, W/16] -> [B, C]
        layer3_s_pool = F.adaptive_avg_pool2d(x_s, (1, 1)).view(x_s.size(0), -1)
        layer3_f_pool = F.adaptive_avg_pool2d(x_f, (1, 1)).view(x_f.size(0), -1)
        
        # 5. Layer4特征处理 [B, 2048, H/32, W/32] -> [B, 2048]
        layer4_pool = F.adaptive_avg_pool2d(x_final, (1, 1)).view(x_final.size(0), -1)
    
        # ==================== 调试信息 ====================
        # print(f"池化后维度:")
        # print(f"  stem_feat: {stem_feat.shape}")
        # print(f"  layer1_s_pool: {layer1_s_pool.shape}")
        # print(f"  layer1_f_pool: {layer1_f_pool.shape}")
        # print(f"  layer2_s_pool: {layer2_s_pool.shape}")
        # print(f"  layer2_f_pool: {layer2_f_pool.shape}")
        # print(f"  layer3_s_pool: {layer3_s_pool.shape}")
        # print(f"  layer3_f_pool: {layer3_f_pool.shape}")
        # print(f"  layer4_pool: {layer4_pool.shape}")
    
        # ==================== 动态初始化投影层 ====================
        if not hasattr(self, 'projection_layers_initialized') or not self.projection_layers_initialized:
            # 获取实际维度
            actual_dims = {
                'stem': stem_feat.size(1),
                'layer1': layer1_s_pool.size(1),
                'layer2': layer2_s_pool.size(1),
                'layer3': layer3_s_pool.size(1),
                'layer4': layer4_pool.size(1)
            }
            # print(f"检测到的实际维度: {actual_dims}")
            
            # 动态初始化投影层
            self._init_dynamic_projections(actual_dims)
            
            # 移动到相应设备
            if stem_feat.is_cuda:
                device = stem_feat.device
                self.stem_proj = self.stem_proj.to(device)
                self.layer1_proj = self.layer1_proj.to(device)
                self.layer2_proj = self.layer2_proj.to(device)
                self.layer3_proj = self.layer3_proj.to(device)
                self.layer4_proj = self.layer4_proj.to(device)
                self.layer_attention = self.layer_attention.to(device)
                self.fusion_processor = self.fusion_processor.to(device)
    
        # ==================== 特征投影到统一维度 ====================
        stem_proj = self.stem_proj(stem_feat)
        
        layer1_s_proj = self.layer1_proj(layer1_s_pool)
        layer1_f_proj = self.layer1_proj(layer1_f_pool)
        
        layer2_s_proj = self.layer2_proj(layer2_s_pool)
        layer2_f_proj = self.layer2_proj(layer2_f_pool)
        
        layer3_s_proj = self.layer3_proj(layer3_s_pool)
        layer3_f_proj = self.layer3_proj(layer3_f_pool)

        # TODO 常规resnet输出
        features['Resnet_feature'] = layer3_s_proj
        
        layer4_proj = self.layer4_proj(layer4_pool)
    
        # print(f"投影后维度: 所有特征都是 [{stem_proj.size(0)}, {self.fusion_dim}]")
    
        # ==================== 注意力加权的特征融合 ====================
        all_layer_features = [
            stem_proj.unsqueeze(1),
            layer1_s_proj.unsqueeze(1),
            layer1_f_proj.unsqueeze(1),
            layer2_s_proj.unsqueeze(1),
            layer2_f_proj.unsqueeze(1),
            layer3_s_proj.unsqueeze(1),
            layer3_f_proj.unsqueeze(1),
            layer4_proj.unsqueeze(1),
        ]
        
        # 拼接所有特征 [B, 8, self.fusion_dim]
        stacked_features = torch.cat(all_layer_features, dim=1)

        # print(f"拼接后的形状 - stacked_features: {stacked_features.shape}")
        
        # 使用注意力机制学习每层特征的权重
        attention_weights = F.softmax(self.layer_attention(stacked_features), dim=1)
        
        # 加权求和 [B, self.fusion_dim]
        attended_features = (stacked_features * attention_weights).sum(dim=1)
        
        # 最终处理
        final_fused = self.fusion_processor(attended_features)

        # print(f"最终形状 - final_fused: {final_fused.shape}")
        
        features['multi_layer_fused'] = final_fused
        features['attention_weights'] = attention_weights.squeeze(-1)
        
        return features

    
    def get_feature_dims(self) -> Dict[str, Tuple[int, ...]]:
        """
        返回各层特征的维度信息，便于后续处理
        
        Returns:
            各层特征的形状字典
        """
        # 这里返回的是各层特征的大致维度，实际维度可能因输入尺寸略有变化
        dims = {
            'stem': (self.inplanes, self.input_resolution//4, self.input_resolution//4),
            'layer1_spatial': (self.inplanes, self.input_resolution//4, self.input_resolution//4),
            'layer1_frequency': (self.inplanes, self.input_resolution//4, self.input_resolution//4),
            'layer2_spatial': (self.inplanes*2, self.input_resolution//8, self.input_resolution//8),
            'layer2_frequency': (self.inplanes*2, self.input_resolution//8, self.input_resolution//8),
            'layer3_spatial': (self.inplanes*4, self.input_resolution//16, self.input_resolution//16),
            'layer3_frequency': (self.inplanes*4, self.input_resolution//16, self.input_resolution//16),
            'layer4': (self.inplanes*8, self.input_resolution//32, self.input_resolution//32),
            'final_pooled': (self.output_dim,)
        }
        return dims


# 构建函数
def create_SFSResnet_extractor(model_name: str = "resnet50", 
                               output_dim: int = 768,
                               use_attention_pool: bool = True,
                               input_resolution: int = 224,
                               **kwargs) -> ModifiedResNet:
    """
    创建特征提取器
    
    Args:
        model_name: 模型变体 ('resnet50', 'resnet101', 'resnet152')
        output_dim: 输出特征维度
        use_attention_pool: 是否使用注意力池化
        **kwargs: ModifiedResNet的额外参数
    
    Returns:
        ModifiedResNet特征提取模型
    """
    model_configs = {
        'resnet50': (3, 4, 6, 3),
        'resnet101': (3, 4, 23, 3),
        'resnet152': (3, 8, 36, 3),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_configs.keys())}")
    
    layers = model_configs[model_name]
    
    model = ModifiedResNet(
        layers=layers,
        output_dim=output_dim,
        heads=8,
        input_resolution=input_resolution,
        width=64,
        use_attention_pool=use_attention_pool,
        **kwargs
    )
    
    return model



        