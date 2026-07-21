from __future__ import annotations

from pathlib import Path

from ...core.config import load_config
from ...core.paths import (
    FORENSIC_MOE_MODEL_DIR,
    FORGELENS_MODEL_DIR,
    LOTA_MODEL_DIR,
    MF2DA_MODEL_DIR,
    TRI_MODEL_DIR,
    UNIVFD_MODEL_DIR,
)
from ...features.trufor.service import build_default_config
from ..app import app


def _status_from_files(model: str, *paths: Path, extra: dict | None = None) -> dict:
    resolved_paths = [Path(path).expanduser().resolve() for path in paths]
    missing = [str(path) for path in resolved_paths if not path.is_file()]
    status = {
        "ready": not missing,
        "model": model,
    }
    if resolved_paths:
        status["checkpoints"] = [str(path) for path in resolved_paths]
        status["checkpoint"] = str(resolved_paths[0])
    if extra:
        status.update(extra)
    if missing:
        status["detail"] = f"Missing required file(s): {', '.join(missing)}"
    return status


@app.get("/api/status")
def status():
    detector_status = {"ready": False, "model": "MARC"}
    audio_status = {"ready": False, "model": "ATADD AST", "model_id": "ast_audioset_ft"}
    try:
        app_config = load_config()
        detector_status = _status_from_files(
            "MARC",
            app_config.checkpoint,
            extra={
                "device": app_config.device,
                "threshold": app_config.threshold,
                "image_size": app_config.image_size,
                "batch_size": app_config.batch_size,
            },
        )

        if app_config.audio is not None:
            audio_status = _status_from_files(
                "ATADD AST",
                app_config.audio.checkpoint,
                extra={
                    "device": app_config.audio.device,
                    "threshold": app_config.audio.threshold,
                    "config_model": app_config.audio.model.name,
                    "sample_rate": app_config.audio.sample_rate,
                    "max_seconds": app_config.audio.max_seconds,
                    "batch_size": app_config.audio.batch_size,
                    "model_id": "ast_audioset_ft",
                },
            )
        else:
            audio_status = {
                "ready": False,
                "model": "ATADD AST",
                "model_id": "ast_audioset_ft",
                "detail": "Audio is not configured in configs/default.json.",
            }
    except Exception as exc:
        detector_status = {"ready": False, "model": "MARC", "detail": str(exc)}
        if "detail" not in audio_status:
            audio_status = {**audio_status, "detail": str(exc)}

    trufor_status = {"ready": False, "model": "TruFor"}
    try:
        trufor_config = build_default_config()
        trufor_status = _status_from_files(
            "TruFor",
            trufor_config.model_file,
            extra={
                "device": trufor_config.device,
                "experiment": trufor_config.experiment,
                "model_file": str(trufor_config.model_file),
            },
        )
    except Exception as exc:
        trufor_status = {"ready": False, "model": "TruFor", "detail": str(exc)}

    image_models = {
        "marc": detector_status,
        "forensic_moe": _status_from_files(
            "Forensic-MoE",
            FORENSIC_MOE_MODEL_DIR / "checkpoints" / "detector.pth",
            FORENSIC_MOE_MODEL_DIR / "checkpoints" / "CLIP.pt",
        ),
        "forgelens": _status_from_files(
            "ForgeLens",
            FORGELENS_MODEL_DIR / "GenImage.pth",
            FORENSIC_MOE_MODEL_DIR / "checkpoints" / "CLIP.pt",
        ),
        "lota": _status_from_files(
            "LOTA",
            LOTA_MODEL_DIR / "lota_weights" / "Network_best.pth",
        ),
        "mf2da": _status_from_files(
            "MF2DA",
            MF2DA_MODEL_DIR / "weights" / "model_epoch_best.pth",
        ),
        "univfd": _status_from_files(
            "UnivFD",
            UNIVFD_MODEL_DIR / "checkpoints" / "fc_weights.pth",
        ),
    }

    video_status = _status_from_files(
        "TRI",
        TRI_MODEL_DIR / "models" / "weights" / "D3.pth",
        TRI_MODEL_DIR / "models" / "weights" / "xclip-base-patch16" / "config.json",
        extra={"threshold": 0.5},
    )

    return {
        "ready": (
            detector_status["ready"]
            or audio_status["ready"]
            or trufor_status["ready"]
            or any(item.get("ready") for item in image_models.values())
            or video_status["ready"]
        ),
        "detector": detector_status,
        "audio": audio_status,
        "trufor": trufor_status,
        "image_models": image_models,
        "audio_models": {
            "ast_audioset_ft": audio_status,
        },
        "video": video_status,
    }
