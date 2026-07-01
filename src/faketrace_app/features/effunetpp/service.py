import base64
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

from ...core.paths import EFFUNETPP_MODEL_DIR
from ...core.uploads import normalize_upload_filename, safe_upload_stem

if str(EFFUNETPP_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(EFFUNETPP_MODEL_DIR))


@dataclass
class EffunetPPLocalizationResult:
    filename: str
    suspicious_ratio: float
    overlay_url: str
    localization_map_url: str
    score: Optional[float]
    saved_files: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def import_runtime():
    try:
        import torch
        import torchvision
        import torch.nn.functional as F
        from PIL import Image
        import os
        import numpy as np
        from net import net
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing dependency for EffUNetPP: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch, torchvision, F, Image, os, np, net


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for EffUNetPP, but torch.cuda.is_available() is false."
        )
    return torch.device(device_name)


def data_url_from_image(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def normalize_map(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim > 2:
        data = np.squeeze(data)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0)


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


class EffunetPPLocalizationEngine:
    def __init__(self, device: str = "auto"):
        (
            self.torch,
            self.torchvision,
            self.F,
            self.Image,
            self.os,
            self.np,
            self.net,
        ) = import_runtime()

        self.device = resolve_device(self.torch, device)
        self.EFFUNETPP_MODEL_DIR = EFFUNETPP_MODEL_DIR
        self.model = self._load_model()

    def _load_model(self):
        model_path = EFFUNETPP_MODEL_DIR / "effunetpp_best_model.pth"

        if not model_path.is_file():
            raise FileNotFoundError(f"EffunetPP model file not found: {model_path}")

        print(f"Loading EffunetPP model from {model_path}")

        # Load the serialized model directly onto the active device.
        model = self.torch.load(model_path, map_location=self.device, weights_only=False)
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_bytes: bytes):
        data_transforms = self.torchvision.transforms.Compose(
            [
                self.torchvision.transforms.Resize((256, 256)),
                self.torchvision.transforms.ToTensor(),
            ]
        )
        image = self.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        input_image = data_transforms(image).unsqueeze(0)

        return input_image, original_size, image

    def predict_uploads(
        self,
        uploads: List[Tuple[str, BinaryIO]],
        save: bool = False,
        output_dir: Path = Path("output"),
    ) -> List[EffunetPPLocalizationResult]:
        results = []

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        for filename, file_obj in uploads:
            normalized_name = normalize_upload_filename(filename)
            content = file_obj.read()
            if not content:
                continue

            image_tensor, original_size, image = self._preprocess_image(content)

            image_tensor = image_tensor.to(self.device)

            with self.torch.no_grad():
                pred, label = self.model(image_tensor)
                pred = self.torch.argmax(pred, dim=1, keepdim=True).float()
                pred = self.F.interpolate(
                    pred,
                    size=(original_size[1], original_size[0]),
                    mode="nearest",
                ).squeeze(0).squeeze(0)

            localization_map = pred.detach().cpu().numpy()
            localization_map = normalize_map(localization_map)
            suspicious_ratio = float((localization_map >= 0.5).mean())
            overlay_img = make_overlay_image(self.Image, image, localization_map)

            localization_img = self.Image.fromarray(to_uint8(localization_map), mode="L")
            score = float(1.0 - self.torch.sigmoid(label).detach().cpu().item())

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

            results.append(
                EffunetPPLocalizationResult(
                    filename=normalized_name,
                    suspicious_ratio=suspicious_ratio,
                    overlay_url=data_url_from_image(overlay_img),
                    localization_map_url=data_url_from_image(localization_img),
                    score=score,
                    saved_files=saved_files,
                )
            )

        return results
