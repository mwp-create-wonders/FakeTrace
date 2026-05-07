from ..app import app
from ..deps import get_detector_engine, get_trufor_engine


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
        "ready": detector_status["ready"] or trufor_status["ready"],
        "detector": detector_status,
        "trufor": trufor_status,
    }
