from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from .config import load_audio_experiment_config
from .dataset import AudioClassificationDataset
from .utils import ensure_dir, save_json


def tensor_stats(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
    }


def run_healthcheck(args: argparse.Namespace) -> dict:
    cfg = load_audio_experiment_config(args.config)
    dataset = AudioClassificationDataset(
        manifest_path=args.manifest,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
        audio_column=cfg.data.audio_column,
        label_column=cfg.data.label_column,
        augment=None,
    )

    missing = []
    label_counts = Counter()
    type_counts = Counter()
    for row in dataset.rows:
        if not row.audio_path.is_file():
            missing.append(str(row.audio_path))
        label_counts[str(row.label)] += 1
        type_counts[str(row.audio_type or "unknown")] += 1

    inspected = []
    max_samples = min(args.max_samples, len(dataset))
    for index in range(max_samples):
        item = dataset[index]
        inspected.append(
            {
                "index": index,
                "source_name": item["source_name"],
                "label": int(item["label"].item()),
                "type": item["type"],
                "input_values": tensor_stats(item["input_values"]),
            }
        )

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "num_rows": len(dataset),
        "sample_rate": cfg.data.sample_rate,
        "max_seconds": cfg.data.max_seconds,
        "expected_num_samples": int(cfg.data.sample_rate * cfg.data.max_seconds),
        "label_counts": dict(label_counts),
        "type_counts": dict(type_counts),
        "missing_count": len(missing),
        "missing_examples": missing[: args.max_missing],
        "inspected": inspected,
    }

    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
        save_json(output_dir / "audio_healthcheck.json", summary)

    print(summary)
    if missing and args.fail_on_missing:
        raise FileNotFoundError(f"{len(missing)} audio files are missing. First missing file: {missing[0]}")
    return summary
