from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset

from .audio_io import load_audio_mono


@dataclass
class AudioManifestRow:
    audio_path: Path
    label: int
    audio_type: str | None = None
    source_name: str | None = None


class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int,
        max_seconds: float,
        audio_column: str = "audio_path",
        label_column: str = "label",
        augment=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_seconds)
        self.audio_column = audio_column
        self.label_column = label_column
        self.augment = augment
        self.rows = self._read_manifest()

    def _read_manifest(self) -> list[AudioManifestRow]:
        rows: list[AudioManifestRow] = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if self.audio_column not in (reader.fieldnames or []):
                raise ValueError(f"Manifest must contain {self.audio_column!r}: {self.manifest_path}")
            if self.label_column not in (reader.fieldnames or []):
                raise ValueError(f"Manifest must contain {self.label_column!r}: {self.manifest_path}")
            for item in reader:
                audio_path = Path(item[self.audio_column])
                if not audio_path.is_absolute():
                    audio_path = self.manifest_path.parent / audio_path
                rows.append(
                    AudioManifestRow(
                        audio_path=audio_path,
                        label=int(item[self.label_column]),
                        audio_type=item.get("type"),
                        source_name=item.get("source_name") or audio_path.name,
                    )
                )
        if not rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        return rows

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
        wav = self._fix_length(wav)
        if self.augment is not None:
            wav = self.augment(wav)
        return {
            "input_values": wav,
            "label": torch.tensor(row.label, dtype=torch.long),
            "type": row.audio_type,
            "source_name": row.source_name or row.audio_path.name,
        }


def collate_audio_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_values": torch.stack([item["input_values"] for item in items], dim=0),
        "labels": torch.stack([item["label"] for item in items], dim=0),
        "types": [item.get("type") for item in items],
        "source_names": [str(item.get("source_name")) for item in items],
    }
