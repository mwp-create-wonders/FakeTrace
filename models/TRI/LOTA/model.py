import torch.nn as nn
import torchvision.models as models
import torch


class model(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        # 使用 MobileNetV3（约 150 万参数，比 EfficientNet-B0 的 500 万小很多）
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrain else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # 冻结大部分层
        if pretrain:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 只解冻最后几层
            for param in self.backbone.features[-3:].parameters():
                param.requires_grad = True
        
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
