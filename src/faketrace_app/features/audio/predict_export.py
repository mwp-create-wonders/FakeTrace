from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audio_io import load_audio_mono
from .config import load_audio_experiment_config
from .experiment import build_model, load_checkpoint_state, resolve_device
from .utils import ensure_dir, save_json


LABEL_TEXT = {0: "real", 1: "fake"}


@dataclass
class PredictRow:
    audio_path: Path
    name: str


class AudioPredictionDataset(Dataset):
    def __init__(self, rows: list[PredictRow], sample_rate: int, max_seconds: float) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_seconds)

    def __len__(self) -> int:
        return len(self.rows)

    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n == self.max_length:
            return wav
        if n > self.max_length:
            return wav[: self.max_length]
        return torch.nn.functional.pad(wav, (0, self.max_length - n))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        wav, sample_rate = load_audio_mono(row.audio_path, torchaudio)
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        return {"input_values": self._fix_length(wav), "name": row.name}


def collate_predict_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_values": torch.stack([item["input_values"] for item in items], dim=0),
        "names": [str(item["name"]) for item in items],
    }


def read_rows_from_manifest(manifest_path: Path) -> list[PredictRow]:
    rows: list[PredictRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "audio_path" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest must contain 'audio_path': {manifest_path}")
        for item in reader:
            audio_path = Path(item["audio_path"])
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            rows.append(PredictRow(audio_path=audio_path, name=item.get("name") or audio_path.name))
    return rows


def read_rows_from_audio_dir(audio_dir: Path, pattern: str) -> list[PredictRow]:
    return [
        PredictRow(audio_path=path, name=path.name)
        for path in sorted(audio_dir.rglob(pattern))
        if path.is_file()
    ]


@torch.no_grad()
def write_predict_stream(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    fake_threshold: float,
    save_probs: bool,
    predict_csv: Path,
    submission_csv: Path,
) -> int:
    model.eval()
    predict_fieldnames = ["name", "predict"]
    if save_probs:
        predict_fieldnames.extend(["prob_real", "prob_fake"])

    num_rows = 0
    with predict_csv.open("w", encoding="utf-8", newline="") as pred_file:
        pred_writer = csv.DictWriter(pred_file, fieldnames=predict_fieldnames)
        pred_writer.writeheader()
        sub_file = None
        sub_writer = None
        if save_probs:
            sub_file = submission_csv.open("w", encoding="utf-8", newline="")
            sub_writer = csv.DictWriter(sub_file, fieldnames=["name", "predict"])
            sub_writer.writeheader()
        try:
            for batch in tqdm(loader, desc="audio-predict"):
                inputs = batch["input_values"].to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                pred_ids = (probs[:, 1] >= fake_threshold).long().tolist()
                probs_cpu = probs.detach().cpu().tolist()
                for name, pred_id, prob_row in zip(batch["names"], pred_ids, probs_cpu):
                    row = {"name": name, "predict": LABEL_TEXT[int(pred_id)]}
                    if save_probs:
                        row["prob_real"] = f"{float(prob_row[0]):.8f}"
                        row["prob_fake"] = f"{float(prob_row[1]):.8f}"
                    pred_writer.writerow(row)
                    if sub_writer is not None:
                        sub_writer.writerow({"name": row["name"], "predict": row["predict"]})
                    num_rows += 1
                pred_file.flush()
                if sub_file is not None:
                    sub_file.flush()
        finally:
            if sub_file is not None:
                sub_file.close()
    return num_rows


def write_submission_zip(zip_path: Path, csv_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(csv_path, arcname="predict.csv")


def resolve_fake_threshold(args: argparse.Namespace) -> tuple[float, str | None]:
    if not args.threshold_summary:
        return float(args.fake_threshold), None

    summary_path = Path(args.threshold_summary).expanduser().resolve()
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    best = summary.get("best")
    if not isinstance(best, dict) or "threshold" not in best:
        raise ValueError(f"Threshold summary does not contain best.threshold: {summary_path}")
    return float(best["threshold"]), str(summary_path)


def run_predict(args: argparse.Namespace) -> dict:
    if bool(args.manifest) == bool(args.audio_dir):
        raise ValueError("Provide exactly one of --manifest or --audio-dir.")

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = load_audio_experiment_config(args.config)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    if args.manifest:
        rows = read_rows_from_manifest(Path(args.manifest))
        input_source = str(Path(args.manifest).resolve())
    else:
        rows = read_rows_from_audio_dir(Path(args.audio_dir), pattern=args.pattern)
        input_source = str(Path(args.audio_dir).resolve())
    if not rows:
        raise ValueError("No audio files found for prediction.")

    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(torch, args.device)
    dataset = AudioPredictionDataset(rows, sample_rate=cfg.data.sample_rate, max_seconds=cfg.data.max_seconds)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_predict_batch,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(cfg, device=device, init_checkpoint=args.checkpoint)
    model.load_state_dict(load_checkpoint_state(args.checkpoint, device, model=model), strict=True)
    fake_threshold, threshold_summary = resolve_fake_threshold(args)

    predict_csv = output_dir / "predict.csv"
    submission_csv = output_dir / "submission_predict.csv" if args.save_probs else predict_csv
    submission_zip = output_dir / "submission.zip"
    num_predictions = write_predict_stream(
        model,
        loader=loader,
        device=device,
        fake_threshold=fake_threshold,
        save_probs=args.save_probs,
        predict_csv=predict_csv,
        submission_csv=submission_csv,
    )
    write_submission_zip(submission_zip, submission_csv)

    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_source": input_source,
        "num_files": num_predictions,
        "predict_csv": str(predict_csv.resolve()),
        "submission_zip": str(submission_zip.resolve()),
        "fake_threshold": fake_threshold,
        "threshold_summary": threshold_summary,
        "save_probs": args.save_probs,
    }
    save_json(output_dir / "predict_summary.json", summary)
    print(summary)
    return summary
