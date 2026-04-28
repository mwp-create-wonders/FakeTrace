import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

from .config import AppConfig
from .paths import TRAINING_DIR, VALID_IMAGE_EXTS


if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


@dataclass
class Prediction:
    filename: str
    path: str
    real_probability: float
    fake_probability: float
    prediction: str

    def to_dict(self) -> dict:
        return asdict(self)


def import_runtime():
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from models.dinov2_models_lora import DINOv2ModelWithLoRA
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing dependency: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "PyTorch failed to load its native DLLs. Reinstall a Windows-compatible torch/torchvision build, "
            "then restart the web app."
        ) from exc

    return torch, Image, transforms, DINOv2ModelWithLoRA


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def collect_image_paths(paths: list[str | Path], recursive: bool = False) -> list[Path]:
    image_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            image_paths.extend(
                item
                for item in path.glob(pattern)
                if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTS
            )
        elif path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            image_paths.append(path)
        elif not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")

    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise ValueError("No image files found.")
    return image_paths


class MARCInferenceEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.torch, self.Image, self.transforms, self.model_cls = import_runtime()
        self.device = resolve_device(self.torch, config.device)
        self.transform = self._build_transform()
        self.model = self._load_model()

    def _build_transform(self):
        return self.transforms.Compose(
            [
                self.transforms.Resize((self.config.image_size, self.config.image_size)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _load_model(self):
        checkpoint_path = self.config.checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_cfg = self.config.model
        model = self.model_cls(
            name=model_cfg.backbone_name,
            binary_num_classes=1,
            source_num_classes=3,
            proj_dim=model_cfg.proj_dim,
            proj_hidden_dim=model_cfg.proj_hidden_dim,
            dropout=model_cfg.dropout,
            lora_rank=model_cfg.lora_rank,
            lora_alpha=model_cfg.lora_alpha,
            lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        )

        checkpoint = self.torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=model_cfg.strict)
        model.to(self.device)
        model.eval()
        return model

    def predict_paths(self, paths: list[Path]) -> list[Prediction]:
        results: list[Prediction] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with self.Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
            results.extend(self._predict_batch(images, batch_paths))
        return results

    def predict_uploads(self, uploads: list[tuple[str, BinaryIO]]) -> list[Prediction]:
        results: list[Prediction] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(uploads), batch_size):
            batch = uploads[start : start + batch_size]
            images = []
            labels = []
            for filename, file_obj in batch:
                image = self.Image.open(file_obj).convert("RGB")
                images.append(self.transform(image))
                labels.append(Path(filename))
            results.extend(self._predict_batch(images, labels))
        return results

    def _predict_batch(self, images: list, labels: list[Path]) -> list[Prediction]:
        tensor = self.torch.stack(images, dim=0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor, return_feature=False, return_tokens=False)
            logits = output["binary_logits"].squeeze(1)
            probs = self.torch.sigmoid(logits).detach().cpu().tolist()

        predictions = []
        for label_path, prob in zip(labels, probs):
            label = "real" if prob >= self.config.threshold else "fake"
            predictions.append(
                Prediction(
                    filename=label_path.name,
                    path=str(label_path),
                    real_probability=float(prob),
                    fake_probability=float(1.0 - prob),
                    prediction=label,
                )
            )
        return predictions
