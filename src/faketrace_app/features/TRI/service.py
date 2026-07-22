import sys
import os
import tempfile
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ...core.paths import TRI_MODEL_DIR

if str(TRI_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(TRI_MODEL_DIR))


@dataclass
class TRIDetectionResult:
    filename: str
    label: str
    probability: float
    prediction: int
    saved_files: dict[str, str] | None = None

    # 视频元数据
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0

    # 帧级证据
    frame_info: list[dict] | None = None
    velocity_l2: list[float] | None = None
    acceleration_l2: list[float] | None = None
    lota_scores: list[float] | None = None

    threshold: float = 0.5
    suspicious_frame_b64: str | None = None
    suspicious_frame_time: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def import_runtime():
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        from models import D3_model
        from Train import AdapterModel1D
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency: {exc.name}") from exc
    return torch, transforms, Image, D3_model, AdapterModel1D


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def normalize_prob(prob: float) -> float:
    return max(0.0, min(1.0, prob))


class TRIDetectionEngine:
    def __init__(self, device: str = "auto"):
        self.torch, self.transforms, self.Image, self.D3_model, self.AdapterModel1D = import_runtime()
        self.device = resolve_device(self.torch, device)

        self.d3_weights_dir = TRI_MODEL_DIR / "weights"
        self.lota_model_path = TRI_MODEL_DIR / "LOTA" / "lota_weights" / "Network_best.pth"
        self.classifier_weights_dir = TRI_MODEL_DIR / "weights"
        self.classifier_file = "l2_residual_independent.pth"
        self.classifier_threshold = 0.5

        self._load_models()

    def _load_models(self):
        print(f"📂 TRI 模型目录: {TRI_MODEL_DIR}")

        print("加载 D3 模型...")
        self.d3_model = self.D3_model(encoder_type="XCLIP-16", loss_type="l2").to(self.device)
        self.d3_model.eval()
        for param in self.d3_model.parameters():
            param.requires_grad = False

        print(f"加载 LOTA 模型: {self.lota_model_path}")
        self.lota_model = self._load_lota_model(self.lota_model_path)

        print(f"加载分类器: {self.classifier_weights_dir / self.classifier_file}")
        self.classifier, self.classifier_threshold = self._load_classifier(
            self.classifier_weights_dir, self.classifier_file
        )
        self.classifier.eval()

    def _load_lota_model(self, model_path: Path):
        import importlib.util
        lota_test_path = TRI_MODEL_DIR / "LOTA" / "test.py"

        if str(TRI_MODEL_DIR / "LOTA") not in sys.path:
            sys.path.insert(0, str(TRI_MODEL_DIR / "LOTA"))

        spec = importlib.util.spec_from_file_location("LOTA_test", lota_test_path)
        lota_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lota_module)
        DeepLearningModel = lota_module.DeepLearningModel

        model = DeepLearningModel()
        if model_path.exists():
            checkpoint = self.torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _load_classifier(self, model_dir: Path, model_file: str):
        model_path = model_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"分类器模型不存在: {model_path}")

        checkpoint = self.torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        if 'adapter.velocity_res.0.weight' in state_dict:
            hidden_dim = state_dict['adapter.velocity_res.0.weight'].shape[0]
        else:
            hidden_dim = 8

        model = self.AdapterModel1D(
            feature_dim=768,
            hidden_dim=hidden_dim,
            kernel_size=3,
            dropout=0.3
        ).to(self.device)

        model.load_state_dict(state_dict)
        best_threshold = checkpoint.get('best_threshold', 0.5)
        self.classifier_threshold = best_threshold  # ← 保存动态阈值
        print(f"  ✅ 加载分类器 (hidden_dim={hidden_dim}, 阈值={best_threshold:.4f})")
        return model, best_threshold

    def _get_video_metadata(self, video_path: str) -> dict:
        """用 OpenCV 提取视频元数据"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'duration': 0.0, 'width': 0, 'height': 0, 'fps': 0.0, 'total_frames': 0}

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0

            cap.release()
            return {'duration': duration, 'width': width, 'height': height, 'fps': fps, 'total_frames': total_frames}
        except Exception:
            return {'duration': 0.0, 'width': 0, 'height': 0, 'fps': 0.0, 'total_frames': 0}

    def _extract_sequence_features(self, video_path: str):
        """
        对齐训练：取第 1 秒 8 帧做差分
        返回速度/加速度各 6 个 + LOTA 最高帧 Base64
        """
        import cv2
        import base64

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # ===== 取第 1 秒 8 帧 =====
        target_frames = 8

        if duration >= 1.0:
            frames_to_take = int(fps)  # 第 1 秒内的总帧数
            indices = np.linspace(0, frames_to_take - 1, target_frames, dtype=int)
        else:
            # 视频不足 1 秒，从全部帧中取 8 帧
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

        # 读取 8 帧
        frame_list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)
        cap.release()

        if len(frame_list) < target_frames:
            return None

        d3_transform = self.transforms.Compose([
            self.transforms.ToTensor(),
            self.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
        ])

        frames_tensor = []
        for frame in frame_list:
            img = self.Image.fromarray(frame)
            frames_tensor.append(d3_transform(img))

        if len(frames_tensor) < 8:
            return None

        frames_tensor = self.torch.stack(frames_tensor).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            output = self.d3_model(frames_tensor, return_vectors=True)
            frame_features = output['frame_features']

        frame_features = frame_features[0].cpu().numpy()

        velocity_raw = frame_features[1:] - frame_features[:-1]
        acceleration_raw = velocity_raw[1:] - velocity_raw[:-1]

        align_len = 6
        velocity_raw = velocity_raw[:align_len]
        acceleration_raw = acceleration_raw[:align_len]

        batch_tensor = frames_tensor.squeeze(0)
        with self.torch.no_grad():
            logits = self.lota_model(batch_tensor)
            lota_scores_full = self.torch.sigmoid(logits).cpu().numpy().flatten()
        lota_scores = lota_scores_full[:align_len]

        suspicious_frame_b64 = None
        suspicious_frame_time = None
        if len(lota_scores_full) >= 8:
            max_idx = int(np.argmax(lota_scores_full[:8]))
            suspicious_frame = frame_list[max_idx]
            frame_bgr = cv2.cvtColor(suspicious_frame, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            suspicious_frame_b64 = base64.b64encode(buf).decode('utf-8')
            suspicious_frame_time = max_idx * (1.0 / 8.0)

        frame_info = []
        for i in range(align_len):
            frame_info.append({
                'frame_number': i + 1,
                'target_time': i * (1.0 / 8.0),
                'actual_time': None,
            })

        return {
            'velocity_raw': velocity_raw,
            'acceleration_raw': acceleration_raw,
            'lota_scores': lota_scores,
            'velocity_l2': np.linalg.norm(velocity_raw, axis=1),
            'acceleration_l2': np.linalg.norm(acceleration_raw, axis=1),
            'frame_info': frame_info,
            'suspicious_frame_b64': suspicious_frame_b64,
            'suspicious_frame_time': suspicious_frame_time,
        }

    def predict_single(self, video_bytes: bytes, filename: str) -> TRIDetectionResult:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            metadata = self._get_video_metadata(tmp_path)
            features = self._extract_sequence_features(tmp_path)

            if features is None:
                return TRIDetectionResult(
                    filename=filename,
                    label="处理失败",
                    probability=0.5,
                    prediction=-1,
                    saved_files=None,
                    duration=metadata['duration'],
                    width=metadata['width'],
                    height=metadata['height'],
                    fps=metadata['fps'],
                    total_frames=metadata['total_frames'],
                    frame_info=None,
                    velocity_l2=None,
                    acceleration_l2=None,
                    lota_scores=None,
                    threshold=self.classifier_threshold,
                    suspicious_frame_b64=None,
                    suspicious_frame_time=None,
                )

            velocity_t = self.torch.FloatTensor(features['velocity_raw']).unsqueeze(0).to(self.device)
            acceleration_t = self.torch.FloatTensor(features['acceleration_raw']).unsqueeze(0).to(self.device)
            lota_t = self.torch.FloatTensor(features['lota_scores']).unsqueeze(0).to(self.device)

            with self.torch.no_grad():
                logits, _ = self.classifier(velocity_t, acceleration_t, lota_t)
                prob = self.torch.sigmoid(logits).item()

            prob = normalize_prob(prob)
            prediction = 1 if prob > self.classifier_threshold else 0
            label = "AI生成" if prediction == 1 else "真实"

            return TRIDetectionResult(
                filename=filename,
                label=label,
                probability=prob,
                prediction=prediction,
                saved_files=None,
                duration=metadata['duration'],
                width=metadata['width'],
                height=metadata['height'],
                fps=metadata['fps'],
                total_frames=metadata['total_frames'],
                frame_info=features['frame_info'],
                velocity_l2=features['velocity_l2'],
                acceleration_l2=features['acceleration_l2'],
                lota_scores=features['lota_scores'],
                threshold=self.classifier_threshold,
                suspicious_frame_b64=features.get('suspicious_frame_b64'),
                suspicious_frame_time=features.get('suspicious_frame_time'),
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def predict_uploads(
        self,
        uploads: list[tuple[str, BinaryIO]],
        save: bool = False,
        output_dir: Path = Path("output")
    ) -> list[TRIDetectionResult]:
        results = []
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        for filename, file_obj in uploads:
            content = file_obj.read()
            if not content:
                continue
            results.append(self.predict_single(content, filename))
        return results


def create_engine(device: str = "auto") -> TRIDetectionEngine:
    return TRIDetectionEngine(device=device)