import base64
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np
import cv2
import pywt

from ...core.paths import FASSA_MODEL_DIR

if str(FASSA_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FASSA_MODEL_DIR))


@dataclass
class FassaLocalizationResult:
    filename: str
    suspicious_ratio: float
    localization_map_url: str
    overlay_url: str
    saved_files: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def import_runtime():
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import cv2
        import pywt
        from forgery_uniformer import ForgeryUniformerSegmentation
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing dependency for Fassa: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch, F, Image, cv2, pywt, ForgeryUniformerSegmentation


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Fassa, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def data_url_from_image(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_map(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim > 2:
        data = np.squeeze(data)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0)


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def make_heatmap_image(Image, normalized: np.ndarray):
    shape = normalized.shape
    r = np.zeros(shape, dtype=np.float32)
    g = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    
    idx = normalized <= 0.25
    b[idx] = 1.0
    g[idx] = 4.0 * normalized[idx]
    
    idx = (normalized > 0.25) & (normalized <= 0.5)
    b[idx] = 1.0 - 4.0 * (normalized[idx] - 0.25)
    g[idx] = 1.0
    
    idx = (normalized > 0.5) & (normalized <= 0.75)
    r[idx] = 4.0 * (normalized[idx] - 0.5)
    g[idx] = 1.0
    
    idx = normalized > 0.75
    r[idx] = 1.0
    g[idx] = 1.0 - 4.0 * (normalized[idx] - 0.75)
    
    rgb = np.stack([to_uint8(r), to_uint8(g), to_uint8(b)], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def make_overlay_image(Image, source_image, normalized: np.ndarray):
    heatmap = make_heatmap_image(Image, normalized).resize(source_image.size)
    alpha_img = Image.fromarray(to_uint8(normalized * 0.6), mode="L")
    alpha = alpha_img.resize(source_image.size)
    
    source_rgba = source_image.convert("RGBA")
    source_rgba.putalpha(180)
    
    heatmap_rgba = heatmap.convert("RGBA")
    heatmap_rgba.putalpha(alpha)
    
    overlay = Image.new("RGBA", source_image.size)
    overlay.paste(source_rgba, (0, 0))
    overlay.alpha_composite(heatmap_rgba)
    
    return overlay.convert("RGB")


class FassaFeatureExtractor:
    WAVELET = 'db1'
    TARGET_SIZE = (1024, 1024)
    FILL_VALUE = 0

    @staticmethod
    def binary_encoding(matrix):
        h, w = matrix.shape
        encode_matrix = np.zeros((h-2, w-2), dtype=np.uint8)
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                   (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = matrix[i, j]
                binary = []
                for dx, dy in offsets:
                    neighbor = matrix[i+dx, j+dy]
                    binary.append('0' if neighbor > center else '1')
                encode_matrix[i-1, j-1] = int(''.join(binary), 2)
        return encode_matrix

    @staticmethod
    def lbp_to_8channels(matrix):
        h, w = matrix.shape
        lbp_channels = np.zeros((h-2, w-2, 8), dtype=np.uint8)
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
                   (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = matrix[i, j]
                for idx, (dx, dy) in enumerate(offsets):
                    neighbor = matrix[i+dx, j+dy]
                    lbp_channels[i-1, j-1, idx] = 0 if neighbor > center else 1
        return lbp_channels

    @staticmethod
    def restore_size(matrix, target_size, fill_value=FILL_VALUE):
        if len(matrix.shape) == 2:
            h, w = matrix.shape
            target_h, target_w = target_size
            restored = np.full((target_h, target_w), fill_value, dtype=matrix.dtype)
            restored[1:-1, 1:-1] = matrix
        elif len(matrix.shape) == 3:
            h, w, c = matrix.shape
            target_h, target_w = target_size
            restored = np.full((target_h, target_w, c), fill_value, dtype=matrix.dtype)
            restored[1:-1, 1:-1, :] = matrix
        else:
            raise ValueError(f"不支持的矩阵维度：{matrix.shape}")
        return restored

    @staticmethod
    def normalize_channels(tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        if max_val - min_val == 0:
            return np.zeros_like(tensor, dtype=np.float32)
        normalized = (tensor - min_val) / (max_val - min_val)
        return normalized.astype(np.float32)

    @classmethod
    def extract_features(cls, img_gray):
        img_resized = cv2.resize(img_gray, cls.TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        img_float = np.float32(img_resized)

        coeffs1 = pywt.dwt2(img_float, cls.WAVELET)
        LL1, (LH1, HL1, HH1) = coeffs1
        A = np.stack([LL1, LH1, HL1, HH1], axis=-1)

        B_list = []
        for i in range(4):
            subband = A[..., i]
            encode_mat = cls.binary_encoding(subband)
            restored_mat = cls.restore_size(encode_mat, (512, 512))
            B_list.append(restored_mat)
        B = np.stack(B_list, axis=-1).astype(np.float32)
        C = A * B

        hvd_subbands = [LH1, HL1, HH1]
        hvd_8ch_list = []
        for subband in hvd_subbands:
            lbp_8ch = cls.lbp_to_8channels(subband)
            restored_8ch = cls.restore_size(lbp_8ch, (512, 512))
            hvd_8ch_list.append(restored_8ch)
        HVD_24ch = np.concatenate(hvd_8ch_list, axis=-1).astype(np.float32)

        D_list = []
        for i in range(4):
            subband_level1 = A[..., i]
            coeffs2 = pywt.dwt2(subband_level1, cls.WAVELET)
            LL2, (LH2, HL2, HH2) = coeffs2
            D_list.extend([LL2, LH2, HL2, HH2])
        D = np.stack(D_list, axis=-1)

        E_list = []
        for i in range(16):
            subband = D[..., i]
            encode_mat = cls.binary_encoding(subband)
            restored_mat = cls.restore_size(encode_mat, (256, 256))
            E_list.append(restored_mat)
        E = np.stack(E_list, axis=-1).astype(np.float32)
        F = D * E

        E_up = cv2.resize(E, (512, 512), interpolation=cv2.INTER_LINEAR)
        F_up = cv2.resize(F, (512, 512), interpolation=cv2.INTER_LINEAR)

        original_40ch = np.concatenate([B, C, E_up, F_up], axis=-1)
        concatenated = np.concatenate([original_40ch, HVD_24ch], axis=-1)
        normalized_tensor = cls.normalize_channels(concatenated)

        return normalized_tensor


class FassaLocalizationEngine:
    def __init__(self, device: str = "auto"):
        (
            self.torch,
            self.F,
            self.Image,
            self.cv2,
            self.pywt,
            self.ForgeryUniformerSegmentation,
        ) = import_runtime()
        
        self.device = resolve_device(self.torch, device)
        self.FASSA_MODEL_DIR = FASSA_MODEL_DIR
        self.model = self._load_model()

    def _load_model(self):
        model_path = FASSA_MODEL_DIR / "fassa_best_model.pth"
        
        if not model_path.is_file():
            raise FileNotFoundError(f"Fassa model file not found: {model_path}")
        
        print(f'Loading Fassa model from {model_path}')
        
        model = self.ForgeryUniformerSegmentation(
            img_size=512,
            num_classes=2,
            resnet_pretrained=False
        )
        
        checkpoint = self.torch.load(model_path, map_location=self.device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_bytes: bytes):
        img = self.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = img.size
        
        img_512 = img.resize((512, 512), self.Image.BILINEAR)
        img_array = np.array(img_512, dtype=np.float32) / 255.0
        
        img_gray = self.cv2.cvtColor(img_array, self.cv2.COLOR_RGB2GRAY)
        features = FassaFeatureExtractor.extract_features(img_gray)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_array - mean) / std
        
        image_tensor = self.torch.tensor(img_normalized.transpose(2, 0, 1), dtype=self.torch.float32)
        feature_tensor = self.torch.tensor(features.transpose(2, 0, 1), dtype=self.torch.float32)
        
        return image_tensor.unsqueeze(0), feature_tensor.unsqueeze(0), img, original_size

    def predict_uploads(
        self, uploads: List[Tuple[str, BinaryIO]], save: bool = False, output_dir: Path = Path("output")
    ) -> List[FassaLocalizationResult]:
        results = []
        
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, file_obj in uploads:
            content = file_obj.read()
            if not content:
                continue
            
            image_tensor, feature_tensor, rgb_image, original_size = self._preprocess_image(content)
            
            image_tensor = image_tensor.to(self.device)
            feature_tensor = feature_tensor.to(self.device)
            
            with self.torch.no_grad():
                pred = self.model(image_tensor, feature_tensor)
                pred = pred[:, 0:1, :, :] if pred.shape[1] > 1 else pred
                pred = self.torch.sigmoid(pred)[0, 0]
                pred = pred.cpu().numpy()
            
            localization_map = normalize_map(pred)
            
            pred_img = self.Image.fromarray(to_uint8(localization_map), mode="L")
            upsampled_pred = pred_img.resize(original_size, self.Image.BILINEAR)
            localization_map = np.array(upsampled_pred) / 255.0
            
            suspicious_ratio = float((localization_map >= 0.5).mean())
            
            localization_img = make_heatmap_image(self.Image, localization_map)
            overlay_img = make_overlay_image(self.Image, rgb_image, localization_map)
            
            saved_files = None
            if save:
                base_name = Path(filename).stem
                saved_files = {}
                
                loc_path = output_dir / f"{base_name}_localization.png"
                localization_img.save(loc_path)
                saved_files["localization_map"] = str(loc_path)
                
                overlay_path = output_dir / f"{base_name}_overlay.png"
                overlay_img.save(overlay_path)
                saved_files["overlay"] = str(overlay_path)
            
            results.append(
                FassaLocalizationResult(
                    filename=Path(filename).name,
                    suspicious_ratio=suspicious_ratio,
                    localization_map_url=data_url_from_image(localization_img),
                    overlay_url=data_url_from_image(overlay_img),
                    saved_files=saved_files,
                )
            )
        
        return results
