import io
from typing import Any, Literal

from fastapi import File, HTTPException, Query, UploadFile

from ..app import app
from ..deps import get_tri_engine
from ...features.video_report.task_store import create_video_task


def _to_jsonable(value: Any) -> Any:
    """numpy 数组/标量 → Python 原生类型，便于 FastAPI 序列化。"""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@app.post("/api/video/predict")
async def predict_video(
    files: list[UploadFile] = File(...),
    model: Literal["tri"] = Query("tri", description="Video detection model to use"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No video files uploaded.")

    uploads = []
    for file in files:
        content = await file.read()
        if not content:
            continue
        uploads.append((file.filename or "upload.mp4", io.BytesIO(content)))

    if not uploads:
        raise HTTPException(status_code=400, detail="Uploaded video files were empty.")

    try:
        if model != "tri":
            raise HTTPException(status_code=400, detail=f"Unknown video detector: {model}")
        engine = get_tri_engine()
        raw_results = engine.predict_uploads(uploads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = []
    for item in raw_results:
        fake_probability = float(getattr(item, "probability", 0.5))
        prediction = "fake" if getattr(item, "prediction", 0) == 1 else "real"
        if getattr(item, "prediction", 0) < 0:
            prediction = "error"
        results.append(
            {
                "filename": item.filename,
                "path": item.filename,
                "real_probability": float(1.0 - fake_probability),
                "fake_probability": fake_probability,
                "prediction": prediction,
                "label": getattr(item, "label", prediction),
                "duration": float(getattr(item, "duration", 0.0)),
                "width": int(getattr(item, "width", 0)),
                "height": int(getattr(item, "height", 0)),
                "fps": float(getattr(item, "fps", 0.0)),
                "total_frames": int(getattr(item, "total_frames", 0)),
                "threshold": float(getattr(item, "threshold", 0.5)),
                "frame_info": _to_jsonable(getattr(item, "frame_info", None)),
                "velocity_l2": _to_jsonable(getattr(item, "velocity_l2", None)),
                "acceleration_l2": _to_jsonable(getattr(item, "acceleration_l2", None)),
                "lota_scores": _to_jsonable(getattr(item, "lota_scores", None)),
                "suspicious_frame_b64": getattr(item, "suspicious_frame_b64", None),
                "suspicious_frame_time": getattr(item, "suspicious_frame_time", None),
            }
        )

    task = create_video_task(model="tri", video_count=len(results))

    return {
        "results": results,
        "video_test_id": task.test_id,
        "meta": {
            "device": str(getattr(engine, "device", "unknown")),
            "model": "TRI",
            "threshold": float(getattr(engine, "classifier_threshold", 0.5)),
        },
    }
