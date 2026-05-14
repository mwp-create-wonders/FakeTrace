import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SACCModule(nn.Module):
    """
    Spatially Adaptive Channel Compression (SACC) Module
    空间自适应通道压缩模块

    核心思想:
    - 不同空间位置具有不同的特征复杂度
    - 简单区域可以用更少通道表示，复杂区域保留更多信息
    - 空间注意力引导自适应压缩率

    创新点:
    1. 空间感知的压缩策略
    2. 可学习的空间-通道压缩映射
    3. 多尺度特征融合处理不同复杂度的区域

    Args:
        in_channels: 输入通道数 (64)
        out_channels: 输出通道数 (16)
        num_scales: 多尺度分支数
    """

    def __init__(self, in_channels=64, out_channels=16, num_scales=3):
        super(SACCModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales

        # 空间复杂度估计器 - 估计每个位置的特征复杂度
        self.complexity_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # 多尺度特征提取分支
        self.scale_branches = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 3 + i * 2  # 3, 5, 7
            padding = kernel_size // 2
            self.scale_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size, padding=padding, groups=in_channels // 8),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(inplace=True),
                )
            )

        # 多尺度融合
        self.scale_fusion = nn.Conv2d((in_channels // 2) * num_scales, in_channels, 1)

        # 空间自适应压缩门控
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 2, 3, padding=1),  # +1 for complexity map
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.Sigmoid()
        )

        # 主要内容分支 - 处理重要区域
        self.content_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 1),
        )

        # 细节恢复分支 - 保留边缘和纹理信息
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

        # 通道注意力 - 动态调整每个输出通道的重要性
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W] = [B, 64, 512, 512]
        Returns:
            out: [B, C_out, H, W] = [B, 16, 512, 512]
        """
        B, C, H, W = x.shape

        # Step 1: 估计空间复杂度
        complexity_map = self.complexity_estimator(x)  # [B, 1, H, W]

        # Step 2: 多尺度特征提取
        scale_features = []
        for branch in self.scale_branches:
            scale_features.append(branch(x))

        # 融合多尺度特征
        multi_scale_feat = torch.cat(scale_features, dim=1)  # [B, 96, H, W]
        fused_feat = self.scale_fusion(multi_scale_feat)  # [B, 64, H, W]

        # Step 3: 空间自适应门控
        gate_input = torch.cat([fused_feat, complexity_map], dim=1)  # [B, 65, H, W]
        spatial_gate = self.spatial_gate(gate_input)  # [B, 16, H, W]

        # Step 4: 内容和细节双分支处理
        content_feat = self.content_branch(fused_feat)  # [B, 16, H, W]
        detail_feat = self.detail_branch(x)  # [B, 16, H, W]

        # Step 5: 根据复杂度自适应融合内容和细节
        # 复杂区域保留更多细节，简单区域主要用内容表示
        adaptive_feat = complexity_map * detail_feat + (1 - complexity_map) * content_feat

        # Step 6: 应用空间门控和通道注意力
        channel_weights = self.channel_attention(adaptive_feat)  # [B, 16, 1, 1]

        out = adaptive_feat * spatial_gate * channel_weights

        return out