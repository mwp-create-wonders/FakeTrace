from __future__ import annotations

import io
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

from ...core.paths import (
    FORENSIC_MOE_MODEL_DIR,
    FORGELENS_MODEL_DIR,
    LOTA_MODEL_DIR,
    MF2DA_MODEL_DIR,
    UNIVFD_MODEL_DIR,
)


@dataclass
class ImagePrediction:
    filename: str
    path: str
    real_probability: float
    fake_probability: float
    prediction: str

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def prepend_sys_path(path: Path):
    path_text = str(path)
    sys.path.insert(0, path_text)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_text)
        except ValueError:
            pass


def purge_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def _predict_label(fake_probability: float, threshold: float = 0.5) -> str:
    return "fake" if fake_probability >= threshold else "real"


class BaseImageDetector:
    model_name = "base"
    threshold = 0.5

    def __init__(self, device: str = "auto"):
        self.device_name = device
        self.torch = None
        self.Image = None
        self.transforms = None
        self.device = None
        self.model = None

    def predict_uploads(self, uploads: list[tuple[str, BinaryIO]]) -> list[ImagePrediction]:
        results: list[ImagePrediction] = []
        for filename, file_obj in uploads:
            image = self.Image.open(file_obj).convert("RGB")
            fake_probability = self.predict_image(image)
            fake_probability = max(0.0, min(1.0, float(fake_probability)))
            label = _predict_label(fake_probability, self.threshold)
            results.append(
                ImagePrediction(
                    filename=Path(filename).name,
                    path=str(Path(filename)),
                    real_probability=float(1.0 - fake_probability),
                    fake_probability=fake_probability,
                    prediction=label,
                )
            )
        return results

    def predict_image(self, image) -> float:
        raise NotImplementedError


class ForensicMoEDetector(BaseImageDetector):
    model_name = "forensic_moe"

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)
        self.checkpoint_path = FORENSIC_MOE_MODEL_DIR / "checkpoints" / "detector.pth"
        self.clip_path = FORENSIC_MOE_MODEL_DIR / "checkpoints" / "CLIP.pt"
        self.torch, self.Image, self.transforms, self.model_cls = self._import_runtime()
        self.device = resolve_device(self.torch, self.device_name)
        self.preprocess = self._build_transform()
        self.model = self._load_model()

    def _import_runtime(self):
        purge_modules(("models",))
        try:
            import torch
            from PIL import Image
            from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
            with prepend_sys_path(FORENSIC_MOE_MODEL_DIR):
                from models.moe_adapter_clip import MoE_Adapter_CLIP
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Missing dependency for Forensic-MoE: {exc.name}") from exc
        except OSError as exc:
            raise RuntimeError("Forensic-MoE runtime failed to load native torch dependencies.") from exc

        transforms = SimpleNamespace(
            Compose=Compose,
            CenterCrop=CenterCrop,
            Resize=Resize,
            ToTensor=ToTensor,
            Normalize=Normalize,
        )
        return torch, Image, transforms, MoE_Adapter_CLIP

    def _build_transform(self):
        def _convert_rgb(image):
            return image.convert("RGB")

        base = self.transforms.Compose(
            [
                self.transforms.CenterCrop((224, 224)),
                _convert_rgb,
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        fallback = self.transforms.Compose(
            [
                self.transforms.Resize(256),
                self.transforms.CenterCrop((224, 224)),
                _convert_rgb,
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        def preprocess(image):
            try:
                return base(image)
            except (ValueError, RuntimeError):
                return fallback(image)

        return preprocess

    def _load_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Forensic-MoE checkpoint not found: {self.checkpoint_path}")
        if not self.clip_path.is_file():
            raise FileNotFoundError(f"Forensic-MoE CLIP checkpoint not found: {self.clip_path}")

        with pushd(FORENSIC_MOE_MODEL_DIR):
            model = self.model_cls(device=self.device)
            state_dict = self.torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image) -> float:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            fake_probability = self.torch.sigmoid(self.model(tensor))[0][0].item()
        return float(fake_probability)


class ForgeLensDetector(BaseImageDetector):
    model_name = "forgelens"

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)
        self.checkpoint_path = FORGELENS_MODEL_DIR / "GenImage.pth"
        self.clip_path = FORENSIC_MOE_MODEL_DIR / "checkpoints" / "CLIP.pt"
        (
            self.torch,
            self.Image,
            self.transforms,
            self.translate_duplicate,
            self.stage1_cls,
            self.stage2_cls,
            self.clip_module,
        ) = self._import_runtime()
        self.device = resolve_device(self.torch, self.device_name)
        self.preprocess = self._build_transform()
        self.model = self._load_model()

    def _import_runtime(self):
        purge_modules(("models", "util"))
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            with prepend_sys_path(FORGELENS_MODEL_DIR):
                from util import translate_duplicate
                from models.network import net_stage1 as stage1_module
                from models.network import net_stage2 as stage2_module
                from models.network.clip import clip as clip_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Missing dependency for ForgeLens: {exc.name}") from exc
        except OSError as exc:
            raise RuntimeError("ForgeLens runtime failed to load native torch dependencies.") from exc

        return (
            torch,
            Image,
            transforms,
            translate_duplicate,
            stage1_module.net_stage1,
            stage2_module.net_stage2,
            clip_module,
        )

    def _patch_clip_loader(self):
        original_load = self.clip_module.load
        clip_path = str(self.clip_path)

        def local_load(_name, device="cpu", jit=False, download_root=None):
            return original_load(clip_path, device=device, jit=jit, download_root=download_root)

        self.clip_module.load = local_load

    def _build_transform(self):
        return self.transforms.Compose(
            [
                self.transforms.Lambda(self.translate_duplicate),
                self.transforms.CenterCrop((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def _load_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"ForgeLens checkpoint not found: {self.checkpoint_path}")
        if not self.clip_path.is_file():
            raise FileNotFoundError(f"Shared CLIP checkpoint not found: {self.clip_path}")

        self._patch_clip_loader()
        state_dict = self.torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = state_dict.get("model_state_dict", state_dict)

        stage2_opt = SimpleNamespace(
            intermediate_model_path="",
            FAFormer_layers=2,
            FAFormer_head=2,
            FAFormer_reduction_factor=1,
        )
        last_error = None
        for builder in (
            lambda: self.stage2_cls(stage2_opt, train=False),
            lambda: self.stage1_cls(),
        ):
            try:
                model = builder()
                model.load_state_dict(state_dict, strict=False)
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Unable to load ForgeLens checkpoint: {last_error}") from last_error

    def predict_image(self, image) -> float:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
            logits = output[0] if isinstance(output, tuple) else output
            fake_probability = self.torch.sigmoid(logits).reshape(-1)[0].item()
        return float(fake_probability)


class LOTADetector(BaseImageDetector):
    model_name = "lota"

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)
        self.checkpoint_path = LOTA_MODEL_DIR / "checkpoints" / "Network_best.pth"
        self.torch, self.Image, self.transforms, self.model_module = self._import_runtime()
        self.device = resolve_device(self.torch, self.device_name)
        self.preprocess = self._build_transform()
        self.model = self._load_model()

    def _import_runtime(self):
        purge_modules(("model",))
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            with prepend_sys_path(LOTA_MODEL_DIR):
                import model as model_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Missing dependency for LOTA: {exc.name}") from exc
        except OSError as exc:
            raise RuntimeError("LOTA runtime failed to load native torch dependencies.") from exc
        return torch, Image, transforms, model_module

    def _build_transform(self):
        return self.transforms.Compose(
            [
                self.transforms.Resize((256, 256)),
                self.transforms.CenterCrop((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"LOTA checkpoint not found: {self.checkpoint_path}")

        model = self.model_module.resnet50(pretrained=False)
        model.fc = self.torch.nn.Linear(2048, 1)
        state_dict = self.torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image) -> float:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            fake_probability = self.torch.sigmoid(self.model(tensor)).reshape(-1)[0].item()
        return float(fake_probability)


class UnivFDDetector(BaseImageDetector):
    model_name = "univfd"

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)
        self.checkpoint_path = UNIVFD_MODEL_DIR / "pretrained_weights" / "Model_best_weights.pth"
        self.clip_path = FORENSIC_MOE_MODEL_DIR / "checkpoints" / "CLIP.pt"
        self.arch = "CLIP:ViT-L/14"
        self.torch, self.Image, self.transforms, self.repo_models, self.clip_models_module = self._import_runtime()
        self.device = resolve_device(self.torch, self.device_name)
        self.preprocess = self._build_transform()
        self.model = self._load_model()

    def _import_runtime(self):
        purge_modules(("models", "options", "data"))
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            with prepend_sys_path(UNIVFD_MODEL_DIR):
                import models as repo_models
                from models import clip_models as clip_models_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Missing dependency for UnivFD: {exc.name}") from exc
        except OSError as exc:
            raise RuntimeError("UnivFD runtime failed to load native torch dependencies.") from exc
        return torch, Image, transforms, repo_models, clip_models_module

    def _patch_clip_loader(self):
        original_load = self.clip_models_module.clip.load
        clip_path = str(self.clip_path)

        def local_load(_name, device="cpu", jit=False, download_root=None):
            return original_load(clip_path, device=device, jit=jit, download_root=download_root)

        self.clip_models_module.clip.load = local_load

    def _build_transform(self):
        return self.transforms.Compose(
            [
                self.transforms.Resize((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def _load_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"UnivFD checkpoint not found: {self.checkpoint_path}")
        if not self.clip_path.is_file():
            raise FileNotFoundError(f"Shared CLIP checkpoint not found: {self.clip_path}")

        self._patch_clip_loader()
        model = self.repo_models.get_model(self.arch)
        state_dict = self.torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if not isinstance(missing, list):
            missing = list(missing)
        if not isinstance(unexpected, list):
            unexpected = list(unexpected)
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image) -> float:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            fake_probability = self.torch.sigmoid(logits).reshape(-1)[0].item()
        return float(fake_probability)


class MF2DADetector(BaseImageDetector):
    model_name = "mf2da"

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)
        self.checkpoint_path = MF2DA_MODEL_DIR / "pretrained_weights" / "model_best.pth"
        self.clip_path = FORENSIC_MOE_MODEL_DIR / "checkpoints" / "CLIP.pt"
        self.arch = "CLIP:ViT-L/14"
        self.torch, self.nn, self.Image, self.transforms, self.repo_models, self.clip_models_module = self._import_runtime()
        self.device = resolve_device(self.torch, self.device_name)
        self.preprocess = self._build_transform()
        self.model = self._load_model()

    def _import_runtime(self):
        purge_modules(("models", "options", "data", "validate", "utils"))
        try:
            import torch
            from torch import nn
            from PIL import Image
            from torchvision import transforms
            with prepend_sys_path(MF2DA_MODEL_DIR):
                import models as repo_models
                from models import clip_models as clip_models_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Missing dependency for MF2DA: {exc.name}") from exc
        except OSError as exc:
            raise RuntimeError("MF2DA runtime failed to load native torch dependencies.") from exc
        return torch, nn, Image, transforms, repo_models, clip_models_module

    def _patch_clip_loader(self):
        original_load = self.clip_models_module.clip.load
        clip_path = str(self.clip_path)

        def local_load(_name, device="cpu", jit=False, download_root=None):
            return original_load(clip_path, device=device, jit=jit, download_root=download_root)

        self.clip_models_module.clip.load = local_load

    def _build_transform(self):
        return self.transforms.Compose(
            [
                self.transforms.Resize((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def _init_fc_from_checkpoint(self, model, model_weights):
        fc0_weight = model_weights.get("fc.0.weight")
        fc3_weight = model_weights.get("fc.3.weight")
        if fc0_weight is None or fc3_weight is None:
            return

        input_dim = fc0_weight.shape[1]
        hidden_dim = fc0_weight.shape[0]
        output_dim = fc3_weight.shape[0]

        model.feature_dim = input_dim
        model.fc = self.nn.Sequential(
            self.nn.Linear(input_dim, hidden_dim),
            self.nn.ReLU(),
            self.nn.Dropout(0.1),
            self.nn.Linear(hidden_dim, output_dim),
        ).to(self.device)

    def _load_model(self):
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"MF2DA checkpoint not found: {self.checkpoint_path}")
        if not self.clip_path.is_file():
            raise FileNotFoundError(f"Shared CLIP checkpoint not found: {self.clip_path}")

        self._patch_clip_loader()
        model = self.repo_models.get_model(self.arch)
        state_dict = self.torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model_weights = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict
        self._init_fc_from_checkpoint(model, model_weights)
        model.load_state_dict(model_weights, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image) -> float:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            fake_probability = self.torch.sigmoid(logits).reshape(-1)[0].item()
        return float(fake_probability)


AVAILABLE_IMAGE_MODELS = {
    "forensic_moe": ForensicMoEDetector,
    "forgelens": ForgeLensDetector,
    "lota": LOTADetector,
    "mf2da": MF2DADetector,
    "univfd": UnivFDDetector,
}
