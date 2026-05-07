import torch.nn as nn
from .dinov2_models import DINOv2Model


class DINOv2ModelWithLoRA(nn.Module):
    def __init__(
        self,
        name,
        binary_num_classes=1,
        source_num_classes=3,
        proj_dim=256,
        proj_hidden_dim=512,
        dropout=0.0,
        lora_rank=8,
        lora_alpha=1.0,
        lora_targets=None,
    ):
        super(DINOv2ModelWithLoRA, self).__init__()

        self.base_model = DINOv2Model(
            name=name,
            binary_num_classes=binary_num_classes,
            source_num_classes=source_num_classes,
            proj_dim=proj_dim,
            proj_hidden_dim=proj_hidden_dim,
            dropout=dropout,
        )

        self.name = name

        try:
            from .lora import apply_lora_to_linear_layers, get_lora_params
        except ImportError:
            raise ImportError("LoRA module not found. Please check your installation.")

        if lora_targets is None:
            lora_targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

        print(f"Adding LoRA to DINOv2 (rank={lora_rank}, alpha={lora_alpha})")
        print(f"LoRA target modules: {lora_targets}")

        # 只对 backbone 加 LoRA，不改任务头
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_targets,
            trainable_orig=False,
        )

        self._get_lora_params = lambda: get_lora_params(self.base_model.model)

    def get_trainable_params(self):
        """
        训练参数 = LoRA 参数 + 各任务头参数
        """
        lora_params = list(self._get_lora_params())

        binary_head_params = list(self.base_model.binary_head.parameters())
        source_head_params = list(self.base_model.source_head.parameters())
        proj_head_params = list(self.base_model.proj_head.parameters())

        total_lora_params = sum(p.numel() for p in lora_params)
        total_head_params = (
            sum(p.numel() for p in binary_head_params)
            + sum(p.numel() for p in source_head_params)
            + sum(p.numel() for p in proj_head_params)
        )

        print(f"Total LoRA params: {total_lora_params}")
        print(f"Total task-head params: {total_head_params}")

        return lora_params + binary_head_params + source_head_params + proj_head_params

    def get_preprocessing_transforms(self, image_size=224):
        return self.base_model.get_preprocessing_transforms(image_size=image_size)

    def extract_features(self, x, return_tokens=False):
        return self.base_model.extract_features(x, return_tokens=return_tokens)

    def forward(self, x, return_feature=False, return_tokens=False):
        return self.base_model(
            x,
            return_feature=return_feature,
            return_tokens=return_tokens,
        )