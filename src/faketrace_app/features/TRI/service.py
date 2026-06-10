import sys
import os
import io
import base64
import tempfile
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ...core.paths import TRI_MODEL_DIR

# 将 TRI 模型目录加入 Python 路径
if str(TRI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(TRI_MODEL_DIR))


@dataclass
class TRIDetectionResult:
    filename: str
    label: str  # "真实" 或 "AI生成"
    probability: float  # AI生成的概率
    prediction: int  # 0=真实, 1=AI生成
    saved_files: dict[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def import_runtime():
    """延迟导入 TRI 模型依赖"""
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        from models import D3_model
        from Train import TemporalFusionNet
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing dependency for TRI detection: {missing}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch, transforms, Image, D3_model, TemporalFusionNet


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def normalize_prob(prob: float) -> float:
    return max(0.0, min(1.0, prob))


class TRIDetectionEngine:
    def __init__(self, device: str = "auto"):
        self.torch, self.transforms, self.Image, self.D3_model, self.TemporalFusionNet = import_runtime()
        self.device = resolve_device(self.torch, device)

        # 模型权重路径
        self.d3_weights_dir = TRI_MODEL_DIR / "weights"
        self.lota_model_path = TRI_MODEL_DIR / "LOTA" / "lota_weights" / "Network_best.pth"
        self.classifier_weights_dir = TRI_MODEL_DIR / "weights"
        self.classifier_file = "vel_lota.pth"
        self.classifier_threshold = 0.4475

        self._load_models()

    def _load_models(self):
        """加载 D3、LOTA、分类器三个模型"""
        print(f"📂 TRI 模型目录: {TRI_MODEL_DIR}")

        # 1. D3 模型（特征提取器）
        print("加载 D3 模型...")
        self.d3_model = self.D3_model(
            encoder_type="XCLIP-16",
            loss_type="l2",
        ).to(self.device)
        self.d3_model.eval()
        for param in self.d3_model.parameters():
            param.requires_grad = False

        # 2. LOTA 模型（辅助特征）
        print(f"加载 LOTA 模型: {self.lota_model_path}")
        self.lota_model = self._load_lota_model(self.lota_model_path)

        # 3. 分类器模型
        print(f"加载分类器: {self.classifier_weights_dir / self.classifier_file}")
        self.classifier, self.classifier_threshold = self._load_classifier(
            self.classifier_weights_dir, self.classifier_file
        )
        self.classifier.eval()

    def _load_lota_model(self, model_path: Path):
        """加载 LOTA 模型"""
        import importlib.util

        lota_test_path = TRI_MODEL_DIR / "LOTA" / "test.py"
        lota_dir = str(TRI_MODEL_DIR / "LOTA")

        if lota_dir not in sys.path:
            sys.path.insert(0, lota_dir)

        spec = importlib.util.spec_from_file_location("LOTA_test", lota_test_path)
        lota_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lota_module)
        DeepLearningModel = lota_module.DeepLearningModel

        model = DeepLearningModel()
        if model_path.exists():
            checkpoint = self.torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _load_classifier(self, model_dir: Path, model_file: str):
        """加载时序融合分类器"""
        model_path = model_dir / model_file
        input_dim = 2  # 速度标量 + LOTA

        model = self.TemporalFusionNet(
            input_dim=input_dim,
            hidden_dim=64,
            kernel_size=3,
            dropout=0.3
        ).to(self.device)

        if model_path.exists():
            checkpoint = self.torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # 处理可能的维度不匹配
                if list(state_dict['conv1.0.weight'].shape)[1] == 3:
                    new_weight = state_dict['conv1.0.weight'][:, :2, :]
                    state_dict['conv1.0.weight'] = new_weight
                model.load_state_dict(state_dict, strict=False)
                best_threshold = checkpoint.get('best_threshold', 0.5)
                print(f"  ✅ 加载分类器: {model_file} (阈值: {best_threshold:.4f})")
                return model, best_threshold
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"  ✅ 加载分类器: {model_file}")
                return model, 0.5
        else:
            raise FileNotFoundError(f"分类器模型不存在: {model_path}")

    def _extract_sequence_features(self, video_bytes: bytes, target_len: int = 8):
        """
        从视频字节流提取特征
        返回 (6,2) 序列数组 [速度标量, LOTA]
        """
        # 保存临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # 加载 test.py 中的工具函数
            import importlib.util
            tri_test_path = TRI_MODEL_DIR / "test.py"
            spec = importlib.util.spec_from_file_location("tri_test", tri_test_path)
            tri_test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tri_test_module)

            VideoFrameExtractor = tri_test_module.VideoFrameExtractor
            extract_sequence_features_from_frames = tri_test_module.extract_sequence_features_from_frames

            # 提取帧
            extractor = VideoFrameExtractor(tmp_path, target_frames=target_len, target_size=224)
            frame_paths = extractor.extract_frames()

            if frame_paths is None:
                return None

            # D3 预处理
            d3_transform = self.transforms.Compose([
                self.transforms.Resize((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
            ])

            # 提取特征
            features = extract_sequence_features_from_frames(
                frame_paths, self.d3_model, self.lota_model, self.device,
                d3_transform, target_len=target_len
            )

            if features is None:
                return None

            # 构建速度 + LOTA 序列
            seq_len = len(features['velocity'])
            sequence = [[features['velocity'][t], features['lota'][t]] for t in range(seq_len)]

            return np.array(sequence, dtype=np.float32)

        finally:
            if 'extractor' in locals():
                extractor.cleanup()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def predict_single(self, video_bytes: bytes, filename: str) -> TRIDetectionResult:
        """检测单个视频"""
        sequence = self._extract_sequence_features(video_bytes)

        if sequence is None:
            return TRIDetectionResult(
                filename=filename,
                label="处理失败",
                probability=0.5,
                prediction=-1,
                saved_files=None
            )

        # 推理
        X = self.torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.classifier(X)
            prob = self.torch.sigmoid(logits).item()

        prob = normalize_prob(prob)
        prediction = 1 if prob > self.classifier_threshold else 0
        label = "AI生成" if prediction == 1 else "真实"

        return TRIDetectionResult(
            filename=filename,
            label=label,
            probability=prob,
            prediction=prediction,
            saved_files=None
        )

    def predict_uploads(
            self,
            uploads: list[tuple[str, BinaryIO]],
            save: bool = False,
            output_dir: Path = Path("output")
    ) -> list[TRIDetectionResult]:
        """批量处理上传的视频文件"""
        results = []

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        for filename, file_obj in uploads:
            content = file_obj.read()
            if not content:
                continue

            result = self.predict_single(content, filename)
            results.append(result)

        return results


def create_engine(device: str = "auto") -> TRIDetectionEngine:
    """创建 TRI 检测引擎实例"""
    return TRIDetectionEngine(device=device)