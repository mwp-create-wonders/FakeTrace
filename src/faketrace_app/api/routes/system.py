from ..app import app
from ..deps import get_audio_engine, get_detector_engine, get_trufor_engine


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

    return {
        "ready": detector_status["ready"] or audio_status["ready"] or trufor_status["ready"],
        "detector": detector_status,
        "audio": audio_status,
        "trufor": trufor_status,
    }
