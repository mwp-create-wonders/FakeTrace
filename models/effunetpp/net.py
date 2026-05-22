import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F


class net(nn.Module):
    """
    编码器：torchvision efficientnet_b0 官方结构
    解码器：U-Net++ (Nested U-Net) 嵌套跳跃连接 + 密集特征融合
    输出：单通道分割掩码（适配二分类）
    """

    def __init__(self):
        # 编码器特征层
        # enc1: 16, 128x128 (level 1)
        # enc2: 24, 64x64   (level 2)
        # enc3: 40, 32x32   (level 3)
        # enc4: 80, 16x16   (level 4)
        # enc5: 112, 16x16  (level 5，瓶颈层)
        # 上采样层
        # up43: 80→40, 16x16→32x32
        # up54: 112→80, 16x16→16x16 (stride=1，尺寸不变)
        # up32: 40→24, 32x32→64x64
        # up21: 24→16, 64x64→128x128
        # up10: 16→16, 128x128→256x256
        # 密集卷积块
        # X40: 瓶颈层enc5(112) → 80
        # X30: enc3(40) → 40; X31: X30 + up43(X40) → 40+40=80 →40
        # X20: enc2(24) →24; X21: X20 + up32(X30) →24+24=48→24; X22: X21 + up32(X31) →24+24=48→24
        # X10: enc1(16) →16; X11: X10 + up21(X20) →16+16=32→16; X12: X11 + up21(X21) →16+16=32→16; X13: X12 + up21(X22) →16+16=32→16
        # 输出层（保持原输出通道数，适配二分类）
        super().__init__()
        # 加载预训练模型
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.enc = efficientnet.features[0:6]

        self.up54 = nn.ConvTranspose2d(112, 80, kernel_size=1, stride=1)
        self.up43 = nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2)
        self.up32 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.up21 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.up10 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)

        self.x40 = self._conv_block(112, 80)
        self.x30 = self._conv_block(40, 40)
        self.x31 = self._conv_block(80, 40)
        self.x20 = self._conv_block(24, 24)
        self.x21 = self._conv_block(48, 24)
        self.x22 = self._conv_block(48, 24)
        self.x10 = self._conv_block(16, 16)
        self.x11 = self._conv_block(32, 16)
        self.x12 = self._conv_block(32, 16)
        self.x13 = self._conv_block(32, 16)

        self.out = nn.Conv2d(16, 2, kernel_size=1)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.thresh = nn.Parameter(torch.tensor([0.0]))
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.thresh = nn.Parameter(torch.tensor([1.0]))

    def _conv_block(self, in_channels, out_channels):
        """保持与EfficientNet一致的激活函数和归一化"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),  # EfficientNet原生激活
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        with torch.no_grad():
            x = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(x)

        # ===================== 编码器前向传播（与原代码一致） =====================
        # 输入: 3×256×256
        x = self.enc[0](x)  # 0层: 3→32, stride=2 → 128×128
        enc1 = self.enc[1](x)  # 1层: 32→16, stride=1 → 128×128 (level 1)
        enc2 = self.enc[2](enc1)  # 2层: 16→24, stride=2 → 64×64 (level 2)
        enc3 = self.enc[3](enc2)  # 3层: 24→40, stride=2 → 32×32 (level 3)
        enc4 = self.enc[4](enc3)  # 4层: 40→80, stride=2 → 16×16 (level 4)
        enc5 = self.enc[5](enc4)  # 5层: 80→112, stride=1 → 16×16 (level 5/瓶颈层)

        # ===================== U-Net++ 解码器（嵌套密集连接） =====================
        # Level 4 (16×16)
        x40 = self.x40(enc5)  # 瓶颈层特征转换

        # Level 3 (32×32)
        x30 = self.x30(enc3)  # 原始enc3特征
        x40_up = self.up43(x40)  # x40上采样到32×32
        x31 = self.x31(torch.cat([x30, x40_up], dim=1))  # 密集拼接: x30 + up(x40)

        # Level 2 (64×64)
        x20 = self.x20(enc2)  # 原始enc2特征
        x30_up = self.up32(x30)  # x30上采样到64×64
        x21 = self.x21(torch.cat([x20, x30_up], dim=1))  # 密集拼接: x20 + up(x30)
        x31_up = self.up32(x31)  # x31上采样到64×64
        x22 = self.x22(torch.cat([x21, x31_up], dim=1))  # 密集拼接: x21 + up(x31)

        # Level 1 (128×128)
        x10 = self.x10(enc1)  # 原始enc1特征
        x20_up = self.up21(x20)  # x20上采样到128×128
        x11 = self.x11(torch.cat([x10, x20_up], dim=1))  # 密集拼接: x10 + up(x20)
        x21_up = self.up21(x21)  # x21上采样到128×128
        x12 = self.x12(torch.cat([x11, x21_up], dim=1))  # 密集拼接: x11 + up(x21)
        x22_up = self.up21(x22)  # x22上采样到128×128
        x13 = self.x13(torch.cat([x12, x22_up], dim=1))  # 密集拼接: x12 + up(x22)

        # 最终上采样到原图尺寸 (256×256)
        out_up = self.up10(x13)
        # 输出分割掩码
        out = self.out(out_up)

        sum = F.softmax(out, dim=1)
        sum = self.avgpool(sum[:, 1])
        sum = sum.squeeze(1)
        label = (-sum + self.thresh) * 10
        return out, label
