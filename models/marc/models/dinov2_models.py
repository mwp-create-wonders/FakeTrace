from pathlib import Path

import torch
import torch.hub
import torch.nn as nn
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TORCH_HUB_DIR = PROJECT_ROOT / ".cache" / "torch" / "hub"


def _configure_torch_hub_dir() -> Path:
    TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_HUB_DIR))
    return TORCH_HUB_DIR


def _resolve_local_hub_repo() -> Path:
    hub_dir = _configure_torch_hub_dir()
    return hub_dir / "facebookresearch_dinov2_main"


TIMM_MODEL_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2",
    "dinov2_vitb14": "vit_base_patch14_dinov2",
    "dinov2_vitl14": "vit_large_patch14_dinov2",
    "dinov2_vitg14": "vit_giant_patch14_dinov2",
}


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

        print(f"Loading DINOv2 offline backbone: {name}")
        self.name = name
        self.feat_dim = CHANNELS[name]
        self.model = self._build_offline_backbone(name)

        # 多头
        self.binary_head = nn.Linear(self.feat_dim, binary_num_classes)   # 默认 1 维，适合 BCEWithLogitsLoss
        self.source_head = nn.Linear(self.feat_dim, source_num_classes)   # 3 类：real / dm / ar
        self.proj_head = ProjectionHead(
            in_dim=self.feat_dim,
            proj_dim=proj_dim,
            hidden_dim=proj_hidden_dim,
            dropout=dropout,
        )

    def _build_offline_backbone(self, name: str):
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency: timm. Install dependencies with: pip install -r requirements.txt"
            ) from exc

        model_name = TIMM_MODEL_NAMES.get(name)
        if model_name is None:
            raise ValueError(f"Unsupported offline DINOv2 backbone: {name}")

        # The finetuned checkpoint stores 518x518 DINOv2 positional embeddings.
        # `dynamic_img_size=True` keeps the pretrained grid while allowing smaller inputs like 336x336.
        model = timm.create_model(
            model_name,
            pretrained=False,
            img_size=518,
            dynamic_img_size=True,
        )

        # Older Meta hub models expose this parameter; the finetuned checkpoint still contains it.
        if not hasattr(model, "mask_token"):
            model.register_parameter(
                "mask_token",
                nn.Parameter(torch.zeros(1, self.feat_dim), requires_grad=False),
            )

        return model

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

            if isinstance(features_dict, dict) and "x_norm_clstoken" in features_dict:
                cls_feature = features_dict["x_norm_clstoken"]
                if return_tokens:
                    patch_tokens = features_dict.get("x_norm_patchtokens", None)
            elif isinstance(features_dict, torch.Tensor):
                if features_dict.ndim != 3:
                    raise ValueError(f"Unexpected tensor shape from forward_features: {tuple(features_dict.shape)}")
                cls_feature = features_dict[:, 0]
                if return_tokens:
                    patch_tokens = features_dict[:, 1:]
            else:
                raise KeyError("DINOv2 forward_features output missing 'x_norm_clstoken'")

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
