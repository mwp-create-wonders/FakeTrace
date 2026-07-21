from __future__ import annotations

from pathlib import Path

import torch


def load_audio_mono(path: str | Path, torchaudio_module=None) -> tuple[torch.Tensor, int]:
    audio_path = Path(path)
    if torchaudio_module is not None:
        try:
            wav, sample_rate = torchaudio_module.load(audio_path)
            return wav.mean(dim=0), int(sample_rate)
        except Exception as torchaudio_error:
            fallback_error = torchaudio_error
        else:
            fallback_error = None
    else:
        fallback_error = None

    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        message = (
            f"Unable to decode audio file {audio_path}. torchaudio backend failed"
            if fallback_error is not None
            else f"Unable to decode audio file {audio_path}"
        )
        raise RuntimeError(
            f"{message}; install soundfile for FLAC/WAV fallback, or convert the file to WAV/FLAC."
        ) from exc

    try:
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as exc:
        if fallback_error is not None:
            raise RuntimeError(
                f"Unable to decode audio file {audio_path} with torchaudio or soundfile. "
                "This environment is most reliable with WAV/FLAC inputs."
            ) from exc
        raise

    wav = torch.from_numpy(data).mean(dim=1)
    return wav, int(sample_rate)
