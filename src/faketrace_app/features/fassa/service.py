import base64
import gc
import io
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np
import cv2
import pywt

from ...core.paths import FASSA_MODEL_DIR


MAX_CONCURRENT_FILES = 10
DEFAULT_BATCH_SIZE = 10
PARALLEL_THRESHOLD = 2

if str(FASSA_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FASSA_MODEL_DIR))

from image_process import (
    binary_encoding,
    lbp_to_8channels,
    restore_size,
    normalize as normalize_tensor,
    WAVELET,
    TARGET_SIZE,
    FILL_VALUE,
)


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


def extract_features_from_gray(img_gray: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img_float = np.float32(img_resized)

    LL1, (LH1, HL1, HH1) = pywt.dwt2(img_float, WAVELET)
    A = np.stack([LL1, LH1, HL1, HH1], axis=-1)

    B = []
    for i in range(4):
        enc = binary_encoding(A[..., i])
        B.append(restore_size(enc, (512, 512)))
    B = np.stack(B, axis=-1).astype(np.float32)
    C = A * B

    hvd = []
    for sub in [LH1, HL1, HH1]:
        lbp = lbp_to_8channels(sub)
        hvd.append(restore_size(lbp, (512, 512)))
    HVD_24 = np.concatenate(hvd, axis=-1).astype(np.float32)

    D_list = []
    for i in range(4):
        LL2, (LH2, HL2, HH2) = pywt.dwt2(A[..., i], WAVELET)
        D_list.extend([LL2, LH2, HL2, HH2])
    D = np.stack(D_list, axis=-1)

    E_list = []
    for i in range(16):
        enc = binary_encoding(D[..., i])
        E_list.append(restore_size(enc, (256, 256)))
    E = np.stack(E_list, axis=-1).astype(np.float32)
    F = D * E

    E_up = cv2.resize(E, (512, 512))
    F_up = cv2.resize(F, (512, 512))

    out = np.concatenate([B, C, E_up, F_up, HVD_24], axis=-1)
    return normalize_tensor(out)


def _extract_features_worker(args):
    idx, img_bytes = args
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_size = img.size
        
        img_512 = img.resize((512, 512), Image.BILINEAR)
        img_array = np.array(img_512, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_array - mean) / std

        img_array_original = np.array(img, dtype=np.float32) / 255.0
        img_gray = cv2.cvtColor(img_array_original, cv2.COLOR_RGB2GRAY)
        features = extract_features_from_gray(img_gray)
        
        return idx, {
            'features': np.ascontiguousarray(features.transpose(2, 0, 1)),
            'img_normalized': np.ascontiguousarray(img_normalized.transpose(2, 0, 1)),
            'rgb_image': img,
            'original_size': original_size
        }
    except Exception as e:
        print(f"Error extracting features for image {idx}: {e}")
        return idx, None


def extract_features_parallel(image_bytes_list: List[bytes]) -> List[Optional[Dict]]:
    n = len(image_bytes_list)
    results: List[Optional[Dict]] = [None] * n

    num_cores = cpu_count() or 1
    print(f"[FeatureExtraction] Total images: {n}, CPU cores: {num_cores}, PARALLEL_THRESHOLD: {PARALLEL_THRESHOLD}")

    t0 = time.time()

    if n < PARALLEL_THRESHOLD:
        print(f"[FeatureExtraction] Using SERIAL mode (n={n} < threshold={PARALLEL_THRESHOLD})")
        for i, img_bytes in enumerate(image_bytes_list):
            idx, data = _extract_features_worker((i, img_bytes))
            results[idx] = data
    else:
        num_workers = min(num_cores, n)
        print(f"[FeatureExtraction] Using PARALLEL mode with {num_workers} workers")
        args_list = [(i, img_bytes) for i, img_bytes in enumerate(image_bytes_list)]

        t_start = time.time()
        with Pool(processes=num_workers) as pool:
            for idx, data in pool.imap(_extract_features_worker, args_list):
                results[idx] = data
        t_parallel = time.time() - t_start
        print(f"[FeatureExtraction] Parallel extraction done in {t_parallel:.3f}s")

    t_total = time.time() - t0
    success = sum(1 for r in results if r is not None)
    print(f"[FeatureExtraction] Total time: {t_total:.3f}s, success: {success}/{n}")

    return results


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
        
        if self.torch.cuda.is_available():
            self.torch.backends.cuda.matmul.allow_tf32 = True
            self.torch.backends.cudnn.allow_tf32 = True
            self.torch.backends.cudnn.benchmark = True
            print(f"Optimizing model for CUDA with TF32 and cudnn benchmark")
            print(f"Model device: {next(model.parameters()).device}")
        
        return model

    def _preprocess_image(self, image_bytes: bytes, precomputed_features: Optional[np.ndarray] = None):
        img = self.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = img.size
        
        img_512 = img.resize((512, 512), self.Image.BILINEAR)
        img_array = np.array(img_512, dtype=np.float32) / 255.0
        
        if precomputed_features is not None:
            features = precomputed_features
        else:
            img_gray = self.cv2.cvtColor(img_array, self.cv2.COLOR_RGB2GRAY)
            features = extract_features_from_gray(img_gray)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_array - mean) / std
        
        image_tensor = self.torch.tensor(img_normalized.transpose(2, 0, 1), dtype=self.torch.float32)
        feature_tensor = self.torch.tensor(features.transpose(2, 0, 1), dtype=self.torch.float32)
        
        return image_tensor.unsqueeze(0), feature_tensor.unsqueeze(0), img, original_size

    def _process_single_result(
        self, filename: str, content: bytes, features: Optional[np.ndarray] = None,
        save: bool = False, output_dir: Path = Path("output")
    ) -> Optional[FassaLocalizationResult]:
        """旧方法：用于没有预处理的场景"""
        image_tensor = feature_tensor = pred = localization_map = rgb_image = None
        try:
            t_preprocess = time.time()
            image_tensor, feature_tensor, rgb_image, original_size = self._preprocess_image(content, features)
            print(f"[Timing] Preprocess {filename}: {time.time() - t_preprocess:.3f}s")
            
            image_tensor = image_tensor.to(self.device)
            feature_tensor = feature_tensor.to(self.device)
            
            t_infer = time.time()
            with self.torch.no_grad():
                model_output = self.model(image_tensor, feature_tensor)
                pred = model_output[0] if isinstance(model_output, tuple) else model_output
                pred = pred[:, 0:1, :, :] if pred.shape[1] > 1 else pred
                pred = self.torch.sigmoid(pred)[0, 0]
                pred = pred.cpu().numpy()
            print(f"[Timing] Inference {filename}: {time.time() - t_infer:.3f}s")
            
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
            
            return FassaLocalizationResult(
                filename=Path(filename).name,
                suspicious_ratio=suspicious_ratio,
                localization_map_url=data_url_from_image(localization_img),
                overlay_url=data_url_from_image(overlay_img),
                saved_files=saved_files,
            )
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return FassaLocalizationResult(
                filename=Path(filename).name,
                suspicious_ratio=0.0,
                localization_map_url="",
                overlay_url="",
                saved_files=None,
            )
        finally:
            del image_tensor, feature_tensor, pred, localization_map, rgb_image
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            gc.collect()

    def _process_single_result_fast(
        self, filename: str, preprocessed_data: Dict, save: bool = False, output_dir: Path = Path("output")
    ) -> Optional[FassaLocalizationResult]:
        """新方法：直接使用预处理好的数据"""
        image_tensor = feature_tensor = pred = localization_map = None
        try:
            t_tensor = time.time()
            img_tensor = self.torch.from_numpy(preprocessed_data['img_normalized'])
            feat_tensor = self.torch.from_numpy(preprocessed_data['features'])
            print(f"[Timing-DEBUG] Create tensors {filename}: {time.time() - t_tensor:.3f}s")
            
            t_to_gpu = time.time()
            image_tensor = img_tensor.unsqueeze(0).to(self.device)
            feature_tensor = feat_tensor.unsqueeze(0).to(self.device)
            print(f"[Timing-DEBUG] To GPU {filename}: {time.time() - t_to_gpu:.3f}s")
            print(f"[Timing] Tensor conversion {filename}: {time.time() - t_tensor:.3f}s")
            
            t_infer = time.time()
            with self.torch.no_grad():
                model_output = self.model(image_tensor, feature_tensor)
                pred = model_output[0] if isinstance(model_output, tuple) else model_output
                pred = pred[:, 0:1, :, :] if pred.shape[1] > 1 else pred
                pred = self.torch.sigmoid(pred)[0, 0]
                pred = pred.cpu().numpy()
            print(f"[Timing] Inference {filename}: {time.time() - t_infer:.3f}s")
            
            localization_map = normalize_map(pred)
            
            rgb_image = preprocessed_data['rgb_image']
            original_size = preprocessed_data['original_size']
            
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
            
            return FassaLocalizationResult(
                filename=Path(filename).name,
                suspicious_ratio=suspicious_ratio,
                localization_map_url=data_url_from_image(localization_img),
                overlay_url=data_url_from_image(overlay_img),
                saved_files=saved_files,
            )
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return FassaLocalizationResult(
                filename=Path(filename).name,
                suspicious_ratio=0.0,
                localization_map_url="",
                overlay_url="",
                saved_files=None,
            )
        finally:
            del image_tensor, feature_tensor, pred, localization_map

    def predict_uploads(
        self, uploads: List[Tuple[str, BinaryIO]], save: bool = False, output_dir: Path = Path("output"),
        batch_size: int = DEFAULT_BATCH_SIZE, max_files: int = MAX_CONCURRENT_FILES
    ) -> List[FassaLocalizationResult]:
        request_start_time = time.time()
        
        if len(uploads) > max_files:
            print(f"Warning: Number of files ({len(uploads)}) exceeds max limit ({max_files}). Processing first {max_files} files.")
            uploads = uploads[:max_files]
        
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        t1 = time.time()
        filenames = []
        image_bytes_list = []
        for filename, file_obj in uploads:
            content = file_obj.read()
            if content:
                filenames.append(filename)
                image_bytes_list.append(content)
        read_time = time.time() - t1
        print(f"[Timing] File reading: {read_time:.3f}s")
        
        total_files = len(filenames)
        print(f"Extracting features for {total_files} images...")
        
        t2 = time.time()
        preprocessed_list = extract_features_parallel(image_bytes_list)
        feature_time = time.time() - t2
        print(f"[Timing] Feature extraction + preprocessing: {feature_time:.3f}s")
        
        print(f"Feature extraction complete. Running model inference...")
        
        t3 = time.time()
        results = []
        for idx, (filename, preprocessed_data) in enumerate(zip(filenames, preprocessed_list)):
            if preprocessed_data is None:
                results.append(
                    FassaLocalizationResult(
                        filename=Path(filename).name,
                        suspicious_ratio=0.0,
                        localization_map_url="",
                        overlay_url="",
                        saved_files=None,
                    )
                )
                continue
            
            result = self._process_single_result_fast(filename, preprocessed_data, save, output_dir)
            results.append(result)
            print(f"Processed {idx + 1}/{total_files}: {filename}")
        
        inference_time = time.time() - t3
        print(f"[Timing] Model inference: {inference_time:.3f}s")
        
        total_time = time.time() - request_start_time
        print(f"[Timing] TOTAL REQUEST TIME: {total_time:.3f}s (read={read_time:.3f}s, feature={feature_time:.3f}s, inference={inference_time:.3f}s)")
        
        return results