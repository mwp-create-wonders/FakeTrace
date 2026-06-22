from ..app import app
from ..deps import (
    get_audio_engine,
    get_detector_engine,
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
        detector = get_detector_engine()
        detector_status = {
            "ready": True,
            "device": str(detector.device),
            "checkpoint": str(detector.config.checkpoint),
            "threshold": detector.config.threshold,
            "image_size": detector.config.image_size,
            "batch_size": detector.config.batch_size,
        }
    except Exception as exc:
        detector_status = {"ready": False, "detail": str(exc)}

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
        trufor = get_trufor_engine()
        trufor_status = {
            "ready": True,
            "device": str(trufor.device),
            "experiment": trufor.config.experiment,
            "model_file": str(trufor.config.model_file),
        }
    except Exception as exc:
        trufor_status = {"ready": False, "detail": str(exc)}

    image_models = {
        "marc": detector_status,
        "forensic_moe": {"ready": False},
        "forgelens": {"ready": False},
        "lota": {"ready": False},
        "mf2da": {"ready": False},
        "univfd": {"ready": False},
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
