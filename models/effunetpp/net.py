import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F


class net(nn.Module):

    class net_mask(nn.Module):

        class _conv_block(nn.Module):

            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                self.residual = None
                if in_channels != out_channels:
                    self.residual = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )
                self.silu = nn.SiLU(inplace=True)

            def forward(self, x):
                residual = x
                x = self.conv(x)
                if self.residual is not None:
                    residual = self.residual(residual)
                x += residual
                x = self.silu(x)
                return x

        def __init__(self):
            super().__init__()
            efficientnet = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.enc = efficientnet.features

            self.up50 = self._upsamble_block(320, 112)
            self.up40 = self._upsamble_block(112, 40)
            self.up41 = self._upsamble_block(112, 40)
            self.up30 = self._upsamble_block(40, 24)
            self.up31 = self._upsamble_block(40, 24)
            self.up32 = self._upsamble_block(40, 24)
            self.up20 = self._upsamble_block(24, 16)
            self.up21 = self._upsamble_block(24, 16)
            self.up22 = self._upsamble_block(24, 16)
            self.up23 = self._upsamble_block(24, 16)
            self.up10 = self._upsamble_block(16, 3)
            self.up11 = self._upsamble_block(16, 3)
            self.up12 = self._upsamble_block(16, 3)
            self.up13 = self._upsamble_block(16, 3)
            self.up14 = self._upsamble_block(16, 3)

            self.x50 = self._conv_block(320, 320)
            self.x40 = self._conv_block(112, 112)
            self.x41 = self._conv_block(224, 112)
            self.x30 = self._conv_block(40, 40)
            self.x31 = self._conv_block(80, 40)
            self.x32 = self._conv_block(120, 40)
            self.x20 = self._conv_block(24, 24)
            self.x21 = self._conv_block(48, 24)
            self.x22 = self._conv_block(72, 24)
            self.x23 = self._conv_block(96, 24)
            self.x10 = self._conv_block(16, 16)
            self.x11 = self._conv_block(32, 16)
            self.x12 = self._conv_block(48, 16)
            self.x13 = self._conv_block(64, 16)
            self.x14 = self._conv_block(80, 16)
            self.x00 = self._conv_block(3, 3)
            self.x01 = self._conv_block(6, 3)
            self.x02 = self._conv_block(9, 3)
            self.x03 = self._conv_block(12, 3)
            self.x04 = self._conv_block(15, 3)
            self.x05 = self._conv_block(18, 3)

            self.out = nn.Conv2d(3, 2, kernel_size=1)

        def _upsamble_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                # self._conv_block(in_channels, out_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )

        # def _conv_block(self, in_channels, out_channels):
        #     return nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.SiLU(inplace=True),
        #         # nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1)),
        #         # nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
        #         # nn.SiLU(inplace=True),
        #     )

        def forward(self, x):
            with torch.no_grad():
                x = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )(x)

            enc0 = x
            enc1 = self.enc[0](enc0)
            enc1 = self.enc[1](enc1)
            enc2 = self.enc[2](enc1)
            enc3 = self.enc[3](enc2)
            enc4 = self.enc[4](enc3)
            enc4 = self.enc[5](enc4)
            enc5 = self.enc[6](enc4)
            enc5 = self.enc[7](enc5)
            # enc5 = self.enc[8](enc5)

            x50 = self.x50(enc5)
            x40 = self.x40(enc4)
            x30 = self.x30(enc3)
            x20 = self.x20(enc2)
            x10 = self.x10(enc1)
            x00 = self.x00(enc0)
            x41 = self.x41(torch.cat([x40, self.up50(x50)], dim=1))
            x31 = self.x31(torch.cat([x30, self.up40(x40)], dim=1))
            x32 = self.x32(torch.cat([x30, x31, self.up41(x41)], dim=1))
            x21 = self.x21(torch.cat([x20, self.up30(x30)], dim=1))
            x22 = self.x22(torch.cat([x20, x21, self.up31(x31)], dim=1))
            x23 = self.x23(torch.cat([x20, x21, x22, self.up32(x32)], dim=1))
            x11 = self.x11(torch.cat([x10, self.up20(x20)], dim=1))
            x12 = self.x12(torch.cat([x10, x11, self.up21(x21)], dim=1))
            x13 = self.x13(torch.cat([x10, x11, x12, self.up22(x22)], dim=1))
            x14 = self.x14(torch.cat([x10, x11, x12, x13, self.up23(x23)], dim=1))
            x01 = self.x01(torch.cat([x00, self.up10(x10)], dim=1))
            x02 = self.x02(torch.cat([x00, x01, self.up11(x11)], dim=1))
            x03 = self.x03(torch.cat([x00, x01, x02, self.up12(x12)], dim=1))
            x04 = self.x04(torch.cat([x00, x01, x02, x03, self.up13(x13)], dim=1))
            x05 = self.x05(torch.cat([x00, x01, x02, x03, x04, self.up14(x14)], dim=1))
            out = self.out(x05)
            return out

    class net_label(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 128)
            self.silu = nn.SiLU(inplace=True)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x):
            label = self.fc1(x)
            label = self.silu(label)
            label = self.fc2(label)
            return label

    def __init__(self):
        super().__init__()
        self.mask = self.net_mask()

        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.label = self.net_label()

    def mask_to_label(self, mask):
        a = torch.argmax(mask, dim=1)
        c = self.softmax(mask)[:, 1]
        a_avg = self.avgpool(c * a)
        a_max = self.maxpool(c * a)
        a_min = -self.maxpool(-c * a)
        a_msq = self.avgpool(c**2 * a)
        # a_min = self.maxpool((1 - c) * (1 - a))
        # a_msq = self.avgpool((1 - c) * (1 - a))
        c_avg = self.avgpool(c)
        c_max = self.maxpool(c)
        c_min = -self.maxpool(-c)
        c_msq = self.avgpool(c**2)
        ans = torch.cat([a_avg, a_max, a_min, a_msq, c_avg, c_max, c_min, c_msq], dim=1)
        ans = ans.squeeze(-1)
        return ans

    def forward(self, x):
        mask = self.mask(x)
        sum = self.mask_to_label(mask)
        label = self.label(sum)
        return mask, label
