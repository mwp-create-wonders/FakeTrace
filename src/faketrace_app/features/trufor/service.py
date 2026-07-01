import base64
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import numpy as np

from ...core.paths import TRUFOR_MODEL_DIR
from ...core.uploads import normalize_upload_filename, safe_upload_stem


if str(TRUFOR_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(TRUFOR_MODEL_DIR))


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass(frozen=True)
class TruForConfig:
    device: str
    experiment: str
    model_file: Path


@dataclass
class LocalizationResult:
    filename: str
    path: str
    score: Optional[float]
    suspicious_ratio: float
    localization_map_url: str
    confidence_map_url: Optional[str]
    overlay_url: str
    saved_files: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_default_config() -> TruForConfig:
    return TruForConfig(
        device="auto",
        experiment="trufor_ph3",
        model_file=(TRUFOR_MODEL_DIR / "pretrained_models" / "trufor.pth.tar").resolve(),
    )


def import_runtime():
    try:
        import torch
        from PIL import Image
        from lib.config import config as base_config
        from lib.utils import get_model
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing dependency for TruFor: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "TruFor runtime failed to load native dependencies. Reinstall a compatible torch stack, then retry."
        ) from exc

    return torch, Image, base_config, get_model


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for TruFor, but torch.cuda.is_available() is false.")
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
    channel_r = to_uint8(normalized)
    channel_g = to_uint8(1.0 - np.abs(normalized - 0.5) * 2.0)
    channel_b = to_uint8(1.0 - normalized)
    rgb = np.stack([channel_r, channel_g, channel_b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def make_overlay_image(Image, source_image, normalized: np.ndarray):
    heatmap = make_heatmap_image(Image, normalized).resize(source_image.size)
    alpha = Image.fromarray(to_uint8(normalized * 0.7), mode="L")
    overlay = source_image.convert("RGBA").copy()
    heatmap_rgba = heatmap.convert("RGBA")
    heatmap_rgba.putalpha(alpha)
    overlay.alpha_composite(heatmap_rgba)
    return overlay.convert("RGB")


class TruForLocalizationEngine:
    def __init__(self, config: Optional[TruForConfig] = None):
        self.config = config or build_default_config()
        self.torch, self.Image, self.base_config, self.get_model = import_runtime()
        self.device = resolve_device(self.torch, self.config.device)
        self.model, self.runtime_config = self._load_model()

    def _load_model(self):
        if not self.config.model_file.is_file():
            raise FileNotFoundError(f"TruFor model file not found: {self.config.model_file}")

        config = self.base_config.clone()
        config.defrost()
        with pushd(TRUFOR_MODEL_DIR):
            config.merge_from_file(str(TRUFOR_MODEL_DIR / "lib" / "config" / f"{self.config.experiment}.yaml"))
        config.merge_from_list(["TEST.MODEL_FILE", str(self.config.model_file)])
        config.freeze()

        model = self.get_model(config)
        checkpoint = self.torch.load(
            self.config.model_file,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model, config

    def _predict_tensor(self, tensor):
        with self.torch.no_grad():
            pred, conf, det, _ = self.model(tensor, save_np=False)

        pred_map = self.torch.squeeze(pred, 0)
        pred_map = self.torch.nn.functional.softmax(pred_map, dim=0)[1].detach().cpu().numpy()

        conf_map = None
        if conf is not None:
            conf_map = self.torch.squeeze(conf, 0)
            conf_map = self.torch.sigmoid(conf_map)[0].detach().cpu().numpy()

        score = None
        if det is not None:
            score = float(self.torch.sigmoid(det).item())

        return pred_map, conf_map, score

    def predict_uploads(
        self, 
        uploads: List[Tuple[str, BinaryIO]],
        save: bool = False,
        output_dir: Path = Path("output")
    ) -> List[LocalizationResult]:
        results: List[LocalizationResult] = []
        
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        for filename, file_obj in uploads:
            normalized_name = normalize_upload_filename(filename)
            with self.Image.open(file_obj) as image:
                rgb_image = image.convert("RGB")
                rgb_array = np.array(rgb_image, dtype=np.float32) / 256.0

            tensor = self.torch.tensor(rgb_array.transpose(2, 0, 1), dtype=self.torch.float32)
            tensor = tensor.unsqueeze(0).to(self.device)
            pred_map, conf_map, score = self._predict_tensor(tensor)

            localization_map = normalize_map(pred_map)
            confidence_map = normalize_map(conf_map) if conf_map is not None else None
            suspicious_ratio = float((localization_map >= 0.5).mean())

            localization_img = make_heatmap_image(self.Image, localization_map)
            overlay_img = make_overlay_image(self.Image, rgb_image, localization_map)
            confidence_img = (
                self.Image.fromarray(to_uint8(confidence_map), mode="L").convert("RGB")
                if confidence_map is not None
                else None
            )

            saved_files = None
            if save:
                base_name = safe_upload_stem(normalized_name)
                saved_files = {}
                
                loc_path = output_dir / f"{base_name}_localization.png"
                localization_img.save(loc_path)
                saved_files["localization_map"] = str(loc_path)
                
                overlay_path = output_dir / f"{base_name}_overlay.png"
                overlay_img.save(overlay_path)
                saved_files["overlay"] = str(overlay_path)
                
                if confidence_img is not None:
                    conf_path = output_dir / f"{base_name}_confidence.png"
                    confidence_img.save(conf_path)
                    saved_files["confidence_map"] = str(conf_path)

            results.append(
                LocalizationResult(
                    filename=normalized_name,
                    path=normalized_name,
                    score=score,
                    suspicious_ratio=suspicious_ratio,
                    localization_map_url=data_url_from_image(localization_img),
                    confidence_map_url=data_url_from_image(confidence_img) if confidence_img else None,
                    overlay_url=data_url_from_image(overlay_img),
                    saved_files=saved_files,
                )
            )
        return results
