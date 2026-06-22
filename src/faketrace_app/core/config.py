import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT


@dataclass(frozen=True)
class ModelConfig:
    backbone_name: str
    lora_rank: int
    lora_alpha: float
    proj_dim: int
    proj_hidden_dim: int
    dropout: float
    strict: bool


@dataclass(frozen=True)
class AudioModelConfig:
    name: str
    pretrained_name: str
    feature_extractor_name: str | None
    kind: str
    freeze_backbone: bool
    dropout: float
    pooling: str
    asp_bottleneck_size: int


@dataclass(frozen=True)
class AudioConfig:
    checkpoint: Path
    device: str
    batch_size: int
    sample_rate: int
    max_seconds: float
    threshold: float
    model: AudioModelConfig


@dataclass(frozen=True)
class AppConfig:
    checkpoint: Path
    device: str
    batch_size: int
    image_size: int
    threshold: float
    model: ModelConfig
    audio: AudioConfig | None = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    base_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    with base_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if overrides:
        raw = _deep_merge(raw, overrides)

    model_raw = raw.get("model", {})
    audio_raw = raw.get("audio")
    audio_config = None
    if isinstance(audio_raw, dict):
        audio_model_raw = audio_raw.get("model", {})
        audio_config = AudioConfig(
            checkpoint=_resolve_project_path(audio_raw["checkpoint"]),
            device=str(audio_raw.get("device", raw.get("device", "auto"))),
            batch_size=int(audio_raw.get("batch_size", raw.get("batch_size", 1))),
            sample_rate=int(audio_raw.get("sample_rate", 16000)),
            max_seconds=float(audio_raw.get("max_seconds", 5.0)),
            threshold=float(audio_raw.get("threshold", 0.5)),
            model=AudioModelConfig(
                name=str(audio_model_raw.get("name", "ast_audioset_ft")),
                pretrained_name=str(
                    audio_model_raw.get(
                        "pretrained_name",
                        "MIT/ast-finetuned-audioset-10-10-0.4593",
                    )
                ),
                feature_extractor_name=(
                    str(audio_model_raw["feature_extractor_name"])
                    if audio_model_raw.get("feature_extractor_name") is not None
                    else None
                ),
                kind=str(audio_model_raw.get("kind", "spectrogram")),
                freeze_backbone=bool(audio_model_raw.get("freeze_backbone", False)),
                dropout=float(audio_model_raw.get("dropout", 0.1)),
                pooling=str(audio_model_raw.get("pooling", "mean")),
                asp_bottleneck_size=int(audio_model_raw.get("asp_bottleneck_size", 128)),
            ),
        )
    return AppConfig(
        checkpoint=_resolve_project_path(raw["checkpoint"]),
        device=str(raw.get("device", "auto")),
        batch_size=int(raw.get("batch_size", 1)),
        image_size=int(raw.get("image_size", 336)),
        threshold=float(raw.get("threshold", 0.5)),
        model=ModelConfig(
            backbone_name=str(model_raw.get("backbone_name", "dinov2_vitl14")),
            lora_rank=int(model_raw.get("lora_rank", 8)),
            lora_alpha=float(model_raw.get("lora_alpha", 1.0)),
            proj_dim=int(model_raw.get("proj_dim", 256)),
            proj_hidden_dim=int(model_raw.get("proj_hidden_dim", 512)),
            dropout=float(model_raw.get("dropout", 0.0)),
            strict=bool(model_raw.get("strict", True)),
        ),
        audio=audio_config,
    )
