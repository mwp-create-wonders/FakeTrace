import sys
import os
import io
import base64
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np

from ...core.paths import CATNET_MODEL_DIR


if str(CATNET_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CATNET_MODEL_DIR))


@dataclass
class CATNetLocalizationResult:
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
        import numpy as np
        from lib import models
        from lib.config import config, update_config
        from Splicing.data.AbstractDataset import AbstractDataset
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing dependency for CAT-Net: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch, F, Image, np, models, config, update_config, AbstractDataset


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for CAT-Net, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def data_url_from_image(Image, image) -> str:
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


class CATNetLocalizationEngine:
    def __init__(self, device: str = "auto"):
        (
            self.torch,
            self.F,
            self.Image,
            self.np,
            self.models,
            self.config,
            self.update_config,
            self.AbstractDataset,
        ) = import_runtime()
        
        self.device = resolve_device(self.torch, device)
        
        config_path = CATNET_MODEL_DIR / "experiments" / "CAT_full.yaml"
        model_path = CATNET_MODEL_DIR / "output" / "best.pth.tar"
        
        args = type('Args', (), {
            'cfg': str(config_path),
            'opts': ['TEST.MODEL_FILE', str(model_path), 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0']
        })
        self.update_config(self.config, args)
        
        self.model = self._load_model()
        
        class DCTExtractor(self.AbstractDataset):
            def __init__(self, DCT_channels=1):
                super().__init__(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=DCT_channels)
                self.DCT_channels = DCT_channels
            
            def get_tamp(self, index):
                pass
        
        self._dct_extractor = DCTExtractor(DCT_channels=1)

    def _load_model(self):
        model = eval('self.models.' + self.config.MODEL.NAME + '.get_seg_model')(self.config)
        
        if self.config.TEST.MODEL_FILE:
            model_state_file = self.config.TEST.MODEL_FILE
        else:
            raise ValueError("CAT-Net model file is not specified.")
        
        print(f'Loading CAT-Net model from {model_state_file}')
        checkpoint = self.torch.load(model_state_file, map_location=self.device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_bytes: bytes):
        temp_jpg = "__temp_upload.jpg"
        
        try:
            img = self.Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(temp_jpg, quality=100, subsampling=0)
            
            DCT_coef, qtables = self._dct_extractor._get_jpeg_info(temp_jpg)
            img_RGB = self.np.array(self.Image.open(temp_jpg).convert("RGB"))
            
            h, w = img_RGB.shape[0], img_RGB.shape[1]
            
            crop_size = ((-(-h // 8)) * 8, (-(-w // 8)) * 8)
            
            if h < crop_size[0] or w < crop_size[1]:
                temp = self.np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
                temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
                img_RGB = temp
                
                max_h = max(crop_size[0], max([DCT_coef[c].shape[0] for c in range(self._dct_extractor.DCT_channels)]))
                max_w = max(crop_size[1], max([DCT_coef[c].shape[1] for c in range(self._dct_extractor.DCT_channels)]))
                for i in range(self._dct_extractor.DCT_channels):
                    temp = self.np.full((max_h, max_w), 0.0)
                    temp[:DCT_coef[i].shape[0], :DCT_coef[i].shape[1]] = DCT_coef[i][:, :]
                    DCT_coef[i] = temp
            
            s_r = 0
            s_c = 0
            img_RGB = img_RGB[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1], :]
            
            for i in range(self._dct_extractor.DCT_channels):
                DCT_coef[i] = DCT_coef[i][s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
            
            t_RGB = (self.torch.tensor(img_RGB.transpose(2, 0, 1), dtype=self.torch.float) - 127.5) / 127.5
            
            T = 20
            t_DCT_coef = self.torch.tensor(DCT_coef, dtype=self.torch.float)
            t_DCT_vol = self.torch.zeros(size=(T + 1, t_DCT_coef.shape[1], t_DCT_coef.shape[2]))
            t_DCT_vol[0] += (t_DCT_coef == 0).float().squeeze()
            for i in range(1, T):
                t_DCT_vol[i] += (t_DCT_coef == i).float().squeeze()
                t_DCT_vol[i] += (t_DCT_coef == -i).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef >= T).float().squeeze()
            t_DCT_vol[T] += (t_DCT_coef <= -T).float().squeeze()
            
            tensor = self.torch.cat([t_RGB, t_DCT_vol])
            qtable_tensor = self.torch.tensor(qtables[:self._dct_extractor.DCT_channels], dtype=self.torch.float)
            
            return tensor.unsqueeze(0), qtable_tensor.unsqueeze(0), img, img_RGB.shape[:2]
        finally:
            if os.path.exists(temp_jpg):
                os.remove(temp_jpg)

    def predict_uploads(
        self, uploads: List[Tuple[str, BinaryIO]], save: bool = False, output_dir: Path = Path("output")
    ) -> List[CATNetLocalizationResult]:
        results = []
        
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, file_obj in uploads:
            content = file_obj.read()
            if not content:
                continue
            
            tensor, qtable_tensor, rgb_image, original_shape = self._preprocess_image(content)
            
            tensor = tensor.to(self.device)
            qtable_tensor = qtable_tensor.to(self.device)
            
            with self.torch.no_grad():
                pred = self.model(tensor, qtable_tensor)
                pred = self.F.softmax(pred, dim=1)[0, 1]
                pred = pred.cpu().numpy()
            
            localization_map = normalize_map(pred)
            
            original_size = rgb_image.size
            pred_img = self.Image.fromarray(to_uint8(localization_map), mode="L")
            upsampled_pred = pred_img.resize(original_size, self.Image.BILINEAR)
            localization_map = self.np.array(upsampled_pred) / 255.0
            
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
                CATNetLocalizationResult(
                    filename=Path(filename).name,
                    suspicious_ratio=suspicious_ratio,
                    localization_map_url=data_url_from_image(self.Image, localization_img),
                    overlay_url=data_url_from_image(self.Image, overlay_img),
                    saved_files=saved_files,
                )
            )
        
        return results
