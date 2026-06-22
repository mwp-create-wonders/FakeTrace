import io
from typing import Literal

from fastapi import File, HTTPException, Query, UploadFile

from ..app import app
from ..deps import get_tri_engine


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
            }
        )

    return {
        "results": results,
        "meta": {
            "device": str(getattr(engine, "device", "unknown")),
            "model": "TRI",
            "threshold": float(getattr(engine, "classifier_threshold", 0.5)),
        },
    }
