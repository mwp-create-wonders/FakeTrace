import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
class AppConfig:
    checkpoint: Path
    device: str
    batch_size: int
    image_size: int
    threshold: float
    model: ModelConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    base_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    with base_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if overrides:
        raw = _deep_merge(raw, overrides)

    model_raw = raw.get("model", {})
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
    )
