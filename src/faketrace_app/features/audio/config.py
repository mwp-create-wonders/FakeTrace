from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioModelConfig:
    name: str
    pretrained_name: str
    freeze_backbone: bool = True
    dropout: float = 0.1
    kind: str = "waveform"
    feature_extractor_name: str | None = None
    pooling: str = "mean"
    asp_bottleneck_size: int = 128


@dataclass
class AudioDataConfig:
    sample_rate: int = 16000
    max_seconds: float = 5.0
    num_classes: int = 2
    audio_column: str = "audio_path"
    label_column: str = "label"


@dataclass
class AudioTrainConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    main_metric: str = "accuracy"
    early_stop_patience: int = 10
    num_workers: int = 2


@dataclass
class AudioAugmentConfig:
    enable: bool = False
    noise_prob: float = 0.5
    noise_scale: float = 0.003
    gain_prob: float = 0.5
    gain_db: float = 6.0


@dataclass
class AudioExperimentConfig:
    model: AudioModelConfig
    data: AudioDataConfig
    train: AudioTrainConfig
    augment: AudioAugmentConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = value
    return output


def load_audio_experiment_config(
    path: str | Path,
    override: dict[str, Any] | None = None,
) -> AudioExperimentConfig:
    raw = _merge(_load_yaml(Path(path)), override)
    return AudioExperimentConfig(
        model=AudioModelConfig(**raw["model"]),
        data=AudioDataConfig(**raw.get("data", {})),
        train=AudioTrainConfig(**raw.get("train", {})),
        augment=AudioAugmentConfig(**raw.get("augment", {})),
    )
