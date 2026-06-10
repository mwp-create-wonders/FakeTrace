import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

from transformers import CLIPVisionModel, XCLIPVisionModel, AutoModel
import torchvision.models as models

# 设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

Transformers = [
    'CLIP-16',
    'CLIP-32',
    'XCLIP-16',
    'XCLIP-32',
    'DINO-base',
    'DINO-large',
]


class D3_model(nn.Module):
    def __init__(self, encoder_type='CLIP-16', loss_type='cos', local_model_dir=None):
        super(D3_model, self).__init__()
        self.loss_type = loss_type
        self.encoder_type = encoder_type

        # ========== 关键修改：自动定位权重目录 ==========
        if local_model_dir is None:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # D3_model.py 在 models/ 下，weights 在 models/TRI/weights/
            # 但为了通用性，我们在 models/ 下创建 weights 目录
            self.local_model_dir = os.path.join(current_dir, "weights")
        else:
            self.local_model_dir = local_model_dir

        # 确保目录存在
        os.makedirs(self.local_model_dir, exist_ok=True)

        print(f"📂 D3 模型权重目录: {os.path.abspath(self.local_model_dir)}")

        # 加载模型
        if encoder_type == 'CLIP-16':
            self.encoder = self._load_model(
                "openai/clip-vit-base-patch16",
                "clip-vit-base-patch16",
                CLIPVisionModel
            )

        elif encoder_type == 'CLIP-32':
            self.encoder = self._load_model(
                "openai/clip-vit-base-patch32",
                "clip-vit-base-patch32",
                CLIPVisionModel
            )

        elif encoder_type == 'XCLIP-16':
            self.encoder = self._load_model(
                "microsoft/xclip-base-patch16",
                "xclip-base-patch16",
                XCLIPVisionModel
            )

        elif encoder_type == 'XCLIP-32':
            self.encoder = self._load_model(
                "microsoft/xclip-base-patch32",
                "xclip-base-patch32",
                XCLIPVisionModel
            )

        elif encoder_type == 'DINO-base':
            self.encoder = self._load_model(
                "facebook/dinov2-base",
                "dinov2-base",
                AutoModel
            )

        elif encoder_type == 'DINO-large':
            self.encoder = self._load_model(
                "facebook/dinov2-large",
                "dinov2-large",
                AutoModel
            )

        elif encoder_type == 'ResNet-18':
            resnet18 = models.resnet18(pretrained=True)
            modules = list(resnet18.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)

        elif encoder_type == 'VGG-16':
            vgg16 = models.vgg16(pretrained=True)
            modules = list(vgg16.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)

        elif encoder_type == 'EfficientNet-b4':
            efficientnet_b4 = models.efficientnet_b4(pretrained=True)
            modules = list(efficientnet_b4.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)

        elif encoder_type == 'MobileNet-v3':
            mobilenetv3 = timm.create_model('mobilenetv3_large_100', pretrained=True)
            modules = list(mobilenetv3.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)

    def _load_model(self, hf_repo_id, local_dir_name, model_class):
        """智能加载模型：优先本地，其次镜像，最后原地址"""
        local_path = os.path.join(self.local_model_dir, local_dir_name)

        print(f"🔍 查找本地模型: {os.path.abspath(local_path)}")

        # 检查本地是否存在模型（必须有 config.json）
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            print(f"📂 从本地加载模型: {local_path}")
            try:
                return model_class.from_pretrained(local_path)
            except Exception as e:
                print(f"⚠️ 本地模型加载失败 ({e})，尝试在线下载...")
        else:
            print(f"⚠️ 本地模型不存在或不完整: {local_path}")

        # 尝试在线下载（使用镜像）
        print(f"🌐 从Hugging Face下载模型: {hf_repo_id}")
        try:
            return model_class.from_pretrained(hf_repo_id, cache_dir=self.local_model_dir)
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            raise

    def forward(self, x, return_stats=True, return_vectors=False):
        """
        参数:
        - return_vectors: 如果为True，返回特征向量序列（用于分组训练）
        """
        b, t, _, h, w = x.shape
        images = x.reshape(-1, 3, h, w)

        # 提取帧特征
        if self.encoder_type in Transformers:
            outputs = self.encoder(images, output_hidden_states=True)
            frame_features = outputs.pooler_output  # [b*t, feature_dim]
        else:
            frame_features = self.encoder(images)
            if len(frame_features.shape) > 2:
                frame_features = frame_features.view(frame_features.size(0), -1)

        feature_dim = frame_features.shape[-1]
        frame_features = frame_features.reshape(b, t, feature_dim)  # [b, t, 768]

        # 计算速度向量（不压缩成标量）
        velocity_vectors = frame_features[:, 1:, :] - frame_features[:, :-1, :]  # [b, t-1, 768]

        # 计算加速度向量
        acceleration_vectors = velocity_vectors[:, 1:, :] - velocity_vectors[:, :-1, :]  # [b, t-2, 768]

        if return_vectors:
            # 返回向量序列用于后续分组训练
            return {
                'velocity_vectors': velocity_vectors,  # [b, 7, 768]
                'acceleration_vectors': acceleration_vectors,  # [b, 6, 768]
                'frame_features': frame_features  # [b, 8, 768]
            }

        if return_stats:
            # 兼容原有逻辑：返回标量统计量
            dis_1st_seq = torch.norm(velocity_vectors, p=2, dim=-1)  # [b, 7]
            dis_2nd_seq = torch.norm(acceleration_vectors, p=2, dim=-1)  # [b, 6]
            # 这里可以继续写原有逻辑
            return dis_1st_seq, dis_2nd_seq
        else:
            # 兼容原有逻辑：返回标量序列
            dis_1st_seq = torch.norm(velocity_vectors, p=2, dim=-1)
            dis_2nd_seq = torch.norm(acceleration_vectors, p=2, dim=-1)
            return dis_1st_seq, dis_2nd_seq