from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from ...core.config import AppConfig, AudioConfig
from ...core.paths import VALID_AUDIO_EXTS
from .audio_io import load_audio_mono


LABEL_TEXT = {0: "real", 1: "fake"}


@dataclass
class AudioPrediction:
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
        import torchaudio
        from torch import nn
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise RuntimeError(
            f"Missing audio dependency: {missing}. Install torchaudio, transformers, and PyYAML."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Audio runtime failed to load native libraries. Reinstall compatible torch/torchaudio builds."
        ) from exc

    return torch, torchaudio, nn, AutoFeatureExtractor, AutoModel, AutoProcessor


def resolve_device(torch, device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for audio detection, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def collect_audio_paths(paths: list[str | Path], recursive: bool = False) -> list[Path]:
    audio_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            audio_paths.extend(
                item
                for item in path.glob(pattern)
                if item.is_file() and item.suffix.lower() in VALID_AUDIO_EXTS
            )
        elif path.is_file() and path.suffix.lower() in VALID_AUDIO_EXTS:
            audio_paths.append(path)
        elif not path.exists():
            raise FileNotFoundError(f"Audio path not found: {path}")

    audio_paths = sorted(set(audio_paths))
    if not audio_paths:
        raise ValueError("No audio files found.")
    return audio_paths


class AudioDeepfakeEngine:
    def __init__(self, config: AppConfig):
        if config.audio is None:
            raise RuntimeError("Audio detection is not configured.")
        self.config = config.audio
        (
            self.torch,
            self.torchaudio,
            self.nn,
            self.auto_feature_extractor,
            self.auto_model,
            self.auto_processor,
        ) = import_runtime()
        self.device = resolve_device(self.torch, self.config.device)
        self.model = self._load_model()

    def _build_pooling_classes(self):
        torch = self.torch
        nn = self.nn

        class AttentiveStatsPooling(nn.Module):
            def __init__(self, hidden_size: int, bottleneck_size: int = 128) -> None:
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, bottleneck_size),
                    nn.Tanh(),
                    nn.Linear(bottleneck_size, 1),
                )

            @property
            def output_size(self) -> int:
                return self.attention[0].in_features * 2

            def forward(self, hidden):
                weights = torch.softmax(self.attention(hidden), dim=1)
                mean = torch.sum(hidden * weights, dim=1)
                variance = torch.sum((hidden - mean.unsqueeze(1)).pow(2) * weights, dim=1)
                std = torch.sqrt(variance.clamp_min(1e-6))
                return torch.cat([mean, std], dim=-1)

        class TimeFrequencyStatsPooling(nn.Module):
            def __init__(
                self,
                hidden_size: int,
                freq_bins: int,
                time_bins: int,
                bottleneck_size: int = 128,
            ) -> None:
                super().__init__()
                self.freq_bins = freq_bins
                self.time_bins = time_bins
                self.expected_patches = freq_bins * time_bins
                self.token_attention = nn.Sequential(
                    nn.Linear(hidden_size, bottleneck_size),
                    nn.Tanh(),
                    nn.Linear(bottleneck_size, 1),
                )
                self.local_attention = nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1)
                self.hidden_size = hidden_size

            @property
            def output_size(self) -> int:
                return self.hidden_size * 4

            def forward(self, hidden):
                if hidden.shape[1] < self.expected_patches:
                    raise ValueError(
                        f"Need at least {self.expected_patches} patch tokens, got {hidden.shape[1]}."
                    )

                patches = hidden[:, -self.expected_patches :, :]
                grid = patches.reshape(
                    hidden.shape[0],
                    self.freq_bins,
                    self.time_bins,
                    self.hidden_size,
                )
                local_scores = self.local_attention(grid.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
                token_scores = self.token_attention(patches)
                scores = token_scores + local_scores

                weights = torch.softmax(scores, dim=1)
                mean = torch.sum(patches * weights, dim=1)
                variance = torch.sum((patches - mean.unsqueeze(1)).pow(2) * weights, dim=1)
                std = torch.sqrt(variance.clamp_min(1e-6))

                score_grid = scores.reshape(hidden.shape[0], self.freq_bins, self.time_bins)
                time_weights = torch.softmax(score_grid.mean(dim=1), dim=1).unsqueeze(-1)
                time_descriptors = grid.mean(dim=1)
                time_context = torch.sum(time_descriptors * time_weights, dim=1)

                freq_weights = torch.softmax(score_grid.mean(dim=2), dim=1).unsqueeze(-1)
                freq_descriptors = grid.mean(dim=2)
                freq_context = torch.sum(freq_descriptors * freq_weights, dim=1)

                return torch.cat([mean, std, time_context, freq_context], dim=-1)

        return AttentiveStatsPooling, TimeFrequencyStatsPooling

    @staticmethod
    def _ast_patch_grid_size(config) -> tuple[int, int]:
        patch_size = int(getattr(config, "patch_size"))
        freq_stride = int(getattr(config, "frequency_stride"))
        time_stride = int(getattr(config, "time_stride"))
        num_mel_bins = int(getattr(config, "num_mel_bins"))
        max_length = int(getattr(config, "max_length"))
        freq_bins = (num_mel_bins - patch_size) // freq_stride + 1
        time_bins = (max_length - patch_size) // time_stride + 1
        return freq_bins, time_bins

    def _build_model_class(self):
        torch = self.torch
        nn = self.nn
        AutoFeatureExtractor = self.auto_feature_extractor
        AutoModel = self.auto_model
        AutoProcessor = self.auto_processor
        AttentiveStatsPooling, TimeFrequencyStatsPooling = self._build_pooling_classes()
        ast_patch_grid_size = self._ast_patch_grid_size

        class AudioBackboneClassifier(nn.Module):
            def __init__(
                self,
                audio_config: AudioConfig,
                num_classes: int = 2,
            ) -> None:
                super().__init__()
                model_cfg = audio_config.model
                self.backbone = AutoModel.from_pretrained(model_cfg.pretrained_name, trust_remote_code=True)
                self.freeze_backbone = model_cfg.freeze_backbone
                if model_cfg.kind == "clap":
                    hidden_size = int(getattr(self.backbone.config, "projection_dim"))
                else:
                    hidden_size = int(self.backbone.config.hidden_size)

                self.pooling_name = model_cfg.pooling
                if model_cfg.pooling == "mean":
                    self.pooling = None
                    classifier_input_size = hidden_size
                elif model_cfg.pooling == "asp":
                    self.pooling = AttentiveStatsPooling(
                        hidden_size,
                        bottleneck_size=model_cfg.asp_bottleneck_size,
                    )
                    classifier_input_size = self.pooling.output_size
                elif model_cfg.pooling == "tf_asp":
                    if model_cfg.kind != "spectrogram":
                        raise ValueError("tf_asp pooling requires kind='spectrogram'.")
                    freq_bins, time_bins = ast_patch_grid_size(self.backbone.config)
                    self.pooling = TimeFrequencyStatsPooling(
                        hidden_size,
                        freq_bins=freq_bins,
                        time_bins=time_bins,
                        bottleneck_size=model_cfg.asp_bottleneck_size,
                    )
                    classifier_input_size = self.pooling.output_size
                else:
                    raise ValueError(f"Unsupported audio pooling head: {model_cfg.pooling!r}")

                self.dropout = nn.Dropout(p=model_cfg.dropout)
                self.classifier = nn.Linear(classifier_input_size, num_classes)

                if self.freeze_backbone:
                    for parameter in self.backbone.parameters():
                        parameter.requires_grad = False

                self.kind = model_cfg.kind
                if model_cfg.kind == "spectrogram":
                    fe_name = model_cfg.feature_extractor_name or model_cfg.pretrained_name
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(fe_name)
                    self.fe_sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000))
                elif model_cfg.kind == "waveform":
                    self.feature_extractor = None
                    self.fe_sampling_rate = None
                elif model_cfg.kind == "clap":
                    fe_name = model_cfg.feature_extractor_name or model_cfg.pretrained_name
                    self.feature_extractor = AutoProcessor.from_pretrained(fe_name)
                    self.fe_sampling_rate = int(
                        getattr(
                            getattr(self.feature_extractor, "feature_extractor", self.feature_extractor),
                            "sampling_rate",
                            48000,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported audio model kind: {model_cfg.kind!r}")

            def extract_features(self, input_values):
                if self.freeze_backbone:
                    self.backbone.eval()
                    with torch.no_grad():
                        return self._extract_features_impl(input_values)
                return self._extract_features_impl(input_values)

            def _extract_features_impl(self, input_values):
                if self.kind == "clap":
                    device = input_values.device
                    wav_list = [w.detach().cpu().numpy() for w in input_values]
                    fe_out = self.feature_extractor(
                        audio=wav_list,
                        sampling_rate=self.fe_sampling_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                    fe_out = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in fe_out.items()
                    }
                    if hasattr(self.backbone, "get_audio_features"):
                        out = self.backbone.get_audio_features(**fe_out)
                        if torch.is_tensor(out):
                            return out.unsqueeze(1)
                        if hasattr(out, "pooler_output") and out.pooler_output is not None:
                            return out.pooler_output.unsqueeze(1)
                        if hasattr(out, "audio_embeds") and out.audio_embeds is not None:
                            return out.audio_embeds.unsqueeze(1)
                        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                            return out.last_hidden_state.mean(dim=1, keepdim=True)
                        raise TypeError("CLAP get_audio_features did not expose audio embeddings.")
                    out = self.backbone(**fe_out)
                    if hasattr(out, "audio_embeds"):
                        return out.audio_embeds.unsqueeze(1)
                    raise TypeError("CLAP backbone did not expose audio embeddings.")

                if self.feature_extractor is not None:
                    device = input_values.device
                    wav_list = [w.detach().cpu().numpy() for w in input_values]
                    fe_out = self.feature_extractor(
                        wav_list,
                        sampling_rate=self.fe_sampling_rate,
                        return_tensors="pt",
                    )
                    backbone_input = fe_out["input_values"].to(device)
                else:
                    backbone_input = input_values

                out = self.backbone(input_values=backbone_input)
                return out.last_hidden_state

            def forward(self, input_values):
                hidden = self.extract_features(input_values)
                if self.pooling is None:
                    pooled = hidden.mean(dim=1)
                else:
                    pooled = self.pooling(hidden)
                return self.classifier(self.dropout(pooled))

        return AudioBackboneClassifier

    def _load_model(self):
        checkpoint_path = self.config.checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Audio checkpoint not found: {checkpoint_path}")

        model_cls = self._build_model_class()
        model = model_cls(self.config).to(self.device)
        checkpoint = self.torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _fix_length(self, wav):
        max_length = int(self.config.sample_rate * self.config.max_seconds)
        n = wav.shape[-1]
        if n == max_length:
            return wav
        if n > max_length:
            return wav[:max_length]
        return self.torch.nn.functional.pad(wav, (0, max_length - n))

    def _load_audio(self, path: Path):
        wav, sample_rate = load_audio_mono(path, self.torchaudio)
        if sample_rate != self.config.sample_rate:
            wav = self.torchaudio.functional.resample(wav, sample_rate, self.config.sample_rate)
        return self._fix_length(wav)

    def predict_paths(self, paths: list[Path]) -> list[AudioPrediction]:
        results: list[AudioPrediction] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            audios = [self._load_audio(path) for path in batch_paths]
            results.extend(self._predict_batch(audios, batch_paths))
        return results

    def predict_uploads(self, uploads: list[tuple[str, BinaryIO]]) -> list[AudioPrediction]:
        temp_paths: list[tuple[Path, str]] = []
        try:
            for filename, file_obj in uploads:
                suffix = Path(filename).suffix or ".wav"
                with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(file_obj.read())
                    temp_paths.append((Path(temp_file.name), filename))

            results = self.predict_paths([path for path, _ in temp_paths])
            for result, (_, filename) in zip(results, temp_paths):
                result.filename = filename
                result.path = filename
            return results
        finally:
            for path, _ in temp_paths:
                path.unlink(missing_ok=True)

    def _predict_batch(self, audios: list, labels: list[Path]) -> list[AudioPrediction]:
        tensor = self.torch.stack(audios, dim=0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            probs = self.torch.softmax(logits, dim=1).detach().cpu()

        predictions = []
        for label_path, prob_row in zip(labels, probs.tolist()):
            fake_probability = float(prob_row[1])
            pred_id = 1 if fake_probability >= self.config.threshold else 0
            predictions.append(
                AudioPrediction(
                    filename=label_path.name,
                    path=str(label_path),
                    real_probability=float(prob_row[0]),
                    fake_probability=fake_probability,
                    prediction=LABEL_TEXT[pred_id],
                )
            )
        return predictions
