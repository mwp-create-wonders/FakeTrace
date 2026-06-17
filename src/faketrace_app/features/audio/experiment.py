from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...core.config import AudioConfig as RuntimeAudioConfig
from ...core.config import AppConfig, ModelConfig
from .augment import WaveformAugment
from .config import AudioExperimentConfig, load_audio_experiment_config
from .dataset import AudioClassificationDataset, collate_audio_batch
from .metrics import classification_metrics
from .service import AudioDeepfakeEngine, import_runtime
from .utils import append_csv_row, ensure_dir, now_seconds, save_json, set_seed


def resolve_device(torch_module, device_name: str):
    if device_name == "auto":
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested for audio experiment, but torch.cuda.is_available() is false.")
    return torch_module.device(device_name)


def build_runtime_config(
    cfg: AudioExperimentConfig,
    checkpoint: str | Path,
    device: str,
    batch_size: int | None = None,
) -> AppConfig:
    return AppConfig(
        checkpoint=Path("unused-image-checkpoint"),
        device=device,
        batch_size=1,
        image_size=336,
        threshold=0.5,
        model=ModelConfig(
            backbone_name="unused",
            lora_rank=0,
            lora_alpha=1.0,
            proj_dim=0,
            proj_hidden_dim=0,
            dropout=0.0,
            strict=True,
        ),
        audio=RuntimeAudioConfig(
            checkpoint=Path(checkpoint).expanduser().resolve(),
            device=device,
            batch_size=batch_size or cfg.train.batch_size,
            sample_rate=cfg.data.sample_rate,
            max_seconds=cfg.data.max_seconds,
            threshold=0.5,
            model=cfg.model,
        ),
    )


def build_model(cfg: AudioExperimentConfig, device, init_checkpoint: str | Path | None = None):
    checkpoint = init_checkpoint or "unused-audio-checkpoint.pt"
    runtime_config = build_runtime_config(cfg, checkpoint=checkpoint, device=str(device))
    engine_shell = object.__new__(AudioDeepfakeEngine)
    engine_shell.config = runtime_config.audio
    (
        engine_shell.torch,
        engine_shell.torchaudio,
        engine_shell.nn,
        engine_shell.auto_feature_extractor,
        engine_shell.auto_model,
        engine_shell.auto_processor,
    ) = import_runtime()
    model_cls = engine_shell._build_model_class()
    return model_cls(runtime_config.audio, num_classes=cfg.data.num_classes).to(device)


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_types = []

    for batch in loader:
        inputs = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * inputs.shape[0]
        all_true.append(labels.cpu().numpy())
        all_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
        all_types.extend(batch.get("types", []))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = classification_metrics(
        y_true,
        y_pred,
        num_classes=num_classes,
        sample_types=all_types if any(item is not None for item in all_types) else None,
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def build_data_loader(
    manifest: str | Path,
    cfg: AudioExperimentConfig,
    *,
    augment=None,
    shuffle: bool = False,
    device=None,
) -> DataLoader:
    dataset = AudioClassificationDataset(
        manifest_path=manifest,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
        audio_column=cfg.data.audio_column,
        label_column=cfg.data.label_column,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=(getattr(device, "type", None) == "cuda"),
    )


def load_checkpoint_state(path: str | Path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state", checkpoint)
    return checkpoint


def run_train(args: argparse.Namespace) -> dict:
    cfg = load_audio_experiment_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.enable_augment:
        cfg.augment.enable = True

    set_seed(args.seed)
    device = resolve_device(torch, args.device)
    output_dir = ensure_dir(args.output_dir)
    log_csv = output_dir / "training_log.csv"
    run_start = now_seconds()

    augment = WaveformAugment(
        enable=cfg.augment.enable,
        noise_prob=cfg.augment.noise_prob,
        noise_scale=cfg.augment.noise_scale,
        gain_prob=cfg.augment.gain_prob,
        gain_db=cfg.augment.gain_db,
    )
    train_loader = build_data_loader(args.train_manifest, cfg, augment=augment, shuffle=True, device=device)
    val_loader = build_data_loader(args.val_manifest, cfg, augment=None, shuffle=False, device=device)
    model = build_model(cfg, device=device)

    init_checkpoint_path = None
    if args.init_checkpoint is not None:
        init_checkpoint_path = str(Path(args.init_checkpoint).resolve())
        checkpoint_state = load_checkpoint_state(args.init_checkpoint, device)
        if args.init_nonstrict:
            current_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            model.load_state_dict(compatible_state, strict=False)
            print(
                f"Initialized {len(compatible_state)} tensors from {init_checkpoint_path}; "
                f"skipped {len(checkpoint_state) - len(compatible_state)} incompatible tensors."
            )
        else:
            model.load_state_dict(checkpoint_state, strict=True)
        print(f"Initialized model weights from {init_checkpoint_path}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    best_metric = -math.inf
    best_epoch = -1
    bad_epochs = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_true = []
        train_pred = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item()) * inputs.shape[0]
            train_true.append(labels.detach().cpu().numpy())
            train_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        train_metrics = classification_metrics(
            np.concatenate(train_true),
            np.concatenate(train_pred),
            num_classes=cfg.data.num_classes,
        )
        train_loss = train_loss_sum / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device=device, num_classes=cfg.data.num_classes)
        score = val_metrics[cfg.train.main_metric]

        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config_path": str(Path(args.config).resolve()),
                    "model_name": cfg.model.name,
                    "pretrained_name": cfg.model.pretrained_name,
                    "num_classes": cfg.data.num_classes,
                    "dropout": cfg.model.dropout,
                    "pooling": cfg.model.pooling,
                    "asp_bottleneck_size": cfg.model.asp_bottleneck_size,
                    "init_checkpoint": init_checkpoint_path,
                },
                output_dir / "best.pt",
            )
        else:
            bad_epochs += 1

        append_csv_row(
            log_csv,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_track2_macro_f1",
                "main_metric",
                "main_metric_value",
            ],
            row={
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "train_macro_f1": f"{train_metrics['macro_f1']:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_accuracy": f"{val_metrics['accuracy']:.6f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                "val_track2_macro_f1": (
                    f"{val_metrics['track2_macro_f1']:.6f}" if "track2_macro_f1" in val_metrics else ""
                ),
                "main_metric": cfg.train.main_metric,
                "main_metric_value": f"{score:.6f}",
            },
        )

        msg = (
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if "track2_macro_f1" in val_metrics:
            msg += f" val_track2_f1={val_metrics['track2_macro_f1']:.4f}"
        print(msg)

        if bad_epochs >= cfg.train.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    elapsed_sec = now_seconds() - run_start
    gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "seed": args.seed,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_metric_name": cfg.train.main_metric,
        "best_metric_value": best_metric,
        "elapsed_seconds": elapsed_sec,
        "peak_gpu_memory_mb": gpu_mem_mb,
        "output_dir": str(output_dir.resolve()),
        "init_checkpoint": init_checkpoint_path,
    }
    save_json(output_dir / "run_summary.json", summary)
    print("Audio training completed.")
    print(summary)
    return summary


def run_eval(args: argparse.Namespace) -> dict:
    cfg = load_audio_experiment_config(args.config)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(torch, args.device)
    loader = build_data_loader(args.manifest, cfg, augment=None, shuffle=False, device=device)
    model = build_model(cfg, device=device, init_checkpoint=args.checkpoint)
    model.load_state_dict(load_checkpoint_state(args.checkpoint, device), strict=True)

    metrics = evaluate_model(model, loader, device=device, num_classes=cfg.data.num_classes)
    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "metrics": metrics,
    }
    save_json(output_dir / "eval_summary.json", summary)
    print(summary)
    return summary
