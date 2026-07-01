from ...core.config import load_config
from ...features.trufor.service import build_default_config
from ..app import app
from ..deps import (
    get_audio_engine,
    get_forensic_moe_engine,
    get_forgelens_engine,
    get_lota_engine,
    get_mf2da_engine,
    get_tri_engine,
    get_trufor_engine,
    get_univfd_engine,
)


@app.get("/api/status")
def status():
    detector_status = {"ready": False}
    try:
        detector_config = load_config()
        detector_status = {
            "ready": detector_config.checkpoint.is_file(),
            "device": detector_config.device,
            "checkpoint": str(detector_config.checkpoint),
            "threshold": detector_config.threshold,
            "image_size": detector_config.image_size,
            "batch_size": detector_config.batch_size,
            "model": "MARC",
        }
        if not detector_status["ready"]:
            detector_status["detail"] = f"Checkpoint not found: {detector_config.checkpoint}"
    except Exception as exc:
        detector_status = {"ready": False, "model": "MARC", "detail": str(exc)}

    audio_status = {"ready": False}
    try:
        audio = get_audio_engine()
        audio_status = {
            "ready": True,
            "device": str(audio.device),
            "checkpoint": str(audio.config.checkpoint),
            "threshold": audio.config.threshold,
            "model": audio.config.model.name,
            "sample_rate": audio.config.sample_rate,
            "max_seconds": audio.config.max_seconds,
            "batch_size": audio.config.batch_size,
        }
    except Exception as exc:
        audio_status = {"ready": False, "detail": str(exc)}

    trufor_status = {"ready": False}
    try:
        trufor_config = build_default_config()
        trufor_status = {
            "ready": trufor_config.model_file.is_file(),
            "device": trufor_config.device,
            "experiment": trufor_config.experiment,
            "model_file": str(trufor_config.model_file),
            "model": "TruFor",
        }
        if not trufor_status["ready"]:
            trufor_status["detail"] = f"Model file not found: {trufor_config.model_file}"
    except Exception as exc:
        trufor_status = {"ready": False, "model": "TruFor", "detail": str(exc)}

    image_models = {
        "marc": detector_status,
        "forensic_moe": {"ready": False, "model": "Forensic-MoE"},
        "forgelens": {"ready": False, "model": "ForgeLens"},
        "lota": {"ready": False, "model": "LOTA"},
        "mf2da": {"ready": False, "model": "MF2DA"},
        "univfd": {"ready": False, "model": "UnivFD"},
    }

    for key, getter, label in (
        ("forensic_moe", get_forensic_moe_engine, "Forensic-MoE"),
        ("forgelens", get_forgelens_engine, "ForgeLens"),
        ("lota", get_lota_engine, "LOTA"),
        ("mf2da", get_mf2da_engine, "MF2DA"),
        ("univfd", get_univfd_engine, "UnivFD"),
    ):
        try:
            engine = getter()
            image_models[key] = {
                "ready": True,
                "model": label,
                "device": str(getattr(engine, "device", "unknown")),
                "checkpoint": str(getattr(engine, "checkpoint_path", "")),
            }
        except Exception as exc:
            image_models[key] = {"ready": False, "model": label, "detail": str(exc)}

    video_status = {"ready": False}
    try:
        tri = get_tri_engine()
        video_status = {
            "ready": True,
            "model": "TRI",
            "device": str(getattr(tri, "device", "unknown")),
            "threshold": float(getattr(tri, "classifier_threshold", 0.5)),
        }
    except Exception as exc:
        video_status = {"ready": False, "model": "TRI", "detail": str(exc)}

    try:
        get_trufor_engine()
    except Exception as exc:
        if "detail" not in trufor_status:
            trufor_status = {**trufor_status, "detail": str(exc)}

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
        "video": video_status,
    }
