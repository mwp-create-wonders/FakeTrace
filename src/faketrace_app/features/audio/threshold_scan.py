from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_audio_experiment_config
from .dataset import AudioClassificationDataset, collate_audio_batch
from .experiment import build_model, load_checkpoint_state, resolve_device
from .metrics import classification_metrics
from .utils import ensure_dir, save_json


def parse_thresholds(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_threshold_grid(args: argparse.Namespace) -> list[float]:
    explicit = parse_thresholds(args.thresholds)
    if explicit:
        return sorted(set(explicit))

    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")
    if args.logspace:
        if args.min_threshold <= 0 or args.max_threshold <= 0:
            raise ValueError("Log-space threshold scan requires positive min/max thresholds.")
        values = np.logspace(
            math.log10(args.min_threshold),
            math.log10(args.max_threshold),
            num=args.steps,
        )
    else:
        values = np.linspace(args.min_threshold, args.max_threshold, num=args.steps)
    return sorted(set(float(value) for value in values))


def describe_threshold_grid(args: argparse.Namespace, thresholds: list[float]) -> dict:
    return {
        "explicit_thresholds": bool(parse_thresholds(args.thresholds)),
        "logspace": bool(args.logspace),
        "min_threshold": min(thresholds) if thresholds else None,
        "max_threshold": max(thresholds) if thresholds else None,
        "steps": len(thresholds),
    }


@torch.no_grad()
def collect_probabilities(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    model.eval()
    all_true = []
    all_fake_prob = []
    all_types = []
    for batch in tqdm(loader, desc="collect-probs"):
        inputs = batch["input_values"].to(device)
        probs = torch.softmax(model(inputs), dim=1).detach().cpu().numpy()
        all_true.append(batch["labels"].numpy())
        all_fake_prob.append(probs[:, 1])
        all_types.extend(batch.get("types", []))
    return np.concatenate(all_true), np.concatenate(all_fake_prob), all_types


def run_threshold_scan(args: argparse.Namespace) -> dict:
    cfg = load_audio_experiment_config(args.config)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(torch, args.device)
    dataset = AudioClassificationDataset(
        manifest_path=args.manifest,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
        audio_column=cfg.data.audio_column,
        label_column=cfg.data.label_column,
        augment=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(cfg, device=device, init_checkpoint=args.checkpoint)
    model.load_state_dict(load_checkpoint_state(args.checkpoint, device, model=model), strict=True)

    y_true, fake_prob, sample_types = collect_probabilities(model, loader, device=device)
    thresholds = build_threshold_grid(args)
    rows = []
    best_row = None
    best_score = -math.inf
    for threshold in thresholds:
        y_pred = (fake_prob >= threshold).astype(np.int64)
        metrics = classification_metrics(
            y_true,
            y_pred,
            num_classes=cfg.data.num_classes,
            sample_types=sample_types if any(item is not None for item in sample_types) else None,
        )
        score = metrics.get(args.metric)
        if score is None:
            raise ValueError(f"Metric {args.metric!r} is unavailable for this manifest.")
        row = {"threshold": threshold, **metrics}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_row = row

    fieldnames = sorted({key for row in rows for key in row.keys()})
    fieldnames.remove("threshold")
    fieldnames = ["threshold", *fieldnames]
    csv_path = output_dir / "threshold_scan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "metric": args.metric,
        "num_thresholds": len(rows),
        "threshold_grid": describe_threshold_grid(args, thresholds),
        "best": best_row,
        "csv": str(csv_path.resolve()),
    }
    save_json(output_dir / "threshold_scan_summary.json", summary)
    print(summary)
    return summary
