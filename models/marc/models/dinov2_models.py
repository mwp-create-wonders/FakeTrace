import torch
import torch.nn as nn
import torch.hub
import os
from torchvision import transforms

hub_repo = "/root/.cache/torch/hub/facebookresearch_dinov2_main"

CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class ProjectionHead(nn.Module):
    """
    用于对比学习的投影头
    """
    def __init__(self, in_dim, proj_dim=256, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = nn.functional.normalize(z, dim=-1)
        return z


class DINOv2Model(nn.Module):
    """
    适配你当前任务的 DINOv2 主干模型：
    - backbone: DINOv2
    - binary_head: 真假分类
    - source_head: 来源分类（real / dm / ar）
    - proj_head: 对比学习投影头
    """

    def __init__(
        self,
        name,
        binary_num_classes=1,
        source_num_classes=3,
        proj_dim=256,
        proj_hidden_dim=512,
        dropout=0.0,
    ):
        super(DINOv2Model, self).__init__()

        if name not in CHANNELS:
            raise ValueError(f"Unsupported DINOv2 model name: {name}")

        print(f"Loading DINOv2 from hub: {name}")
        self.name = name
        self.feat_dim = CHANNELS[name]

        if os.path.exists(hub_repo):
            print(f"Loading DINOv2 from local hub cache: {hub_repo}")
            self.model = torch.hub.load(hub_repo, name, source="local", pretrained=True)
        else:
            print(f"Loading DINOv2 from hub: {name}")
            self.model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)

        # 多头
        self.binary_head = nn.Linear(self.feat_dim, binary_num_classes)   # 默认 1 维，适合 BCEWithLogitsLoss
        self.source_head = nn.Linear(self.feat_dim, source_num_classes)   # 3 类：real / dm / ar
        self.proj_head = ProjectionHead(
            in_dim=self.feat_dim,
            proj_dim=proj_dim,
            hidden_dim=proj_hidden_dim,
            dropout=dropout,
        )

    def get_preprocessing_transforms(self, image_size=224):
        """
        提供一个默认预处理，方便外部调用
        """
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def extract_features(self, x, return_tokens=False):
        """
        提取 backbone 特征
        return:
            cls_feature: [B, C]
            patch_tokens: [B, N, C] or None
        """
        patch_tokens = None

        if hasattr(self.model, "forward_features"):
            features_dict = self.model.forward_features(x)

            if "x_norm_clstoken" in features_dict:
                cls_feature = features_dict["x_norm_clstoken"]
            else:
                raise KeyError("DINOv2 forward_features output missing 'x_norm_clstoken'")

            if return_tokens:
                # DINOv2 常见字段名是 x_norm_patchtokens
                patch_tokens = features_dict.get("x_norm_patchtokens", None)

        else:
            # 兜底逻辑
            out = self.model(x)
            if isinstance(out, dict):
                if "x_norm_clstoken" in out:
                    cls_feature = out["x_norm_clstoken"]
                elif "last_hidden_state" in out:
                    cls_feature = out["last_hidden_state"][:, 0]
                    if return_tokens:
                        patch_tokens = out["last_hidden_state"][:, 1:]
                else:
                    raise KeyError("Unknown feature dict structure returned by model")
            else:
                cls_feature = out

        return cls_feature, patch_tokens

    def forward(
        self,
        x,
        return_feature=False,
        return_tokens=False,
    ):
        """
        返回统一字典，便于多损失训练
        """
        features, patch_tokens = self.extract_features(x, return_tokens=return_tokens)

        binary_logits = self.binary_head(features)
        source_logits = self.source_head(features)
        proj = self.proj_head(features)

        output = {
            "binary_logits": binary_logits,   # [B, 1] 默认
            "source_logits": source_logits,   # [B, 3]
            "proj": proj,                     # [B, proj_dim]
        }

        if return_feature:
            output["feature"] = features      # [B, C]

        if return_tokens:
            output["tokens"] = patch_tokens   # [B, N, C] or None

        return output