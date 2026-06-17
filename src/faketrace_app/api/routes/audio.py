import io

from fastapi import File, HTTPException, UploadFile

from ..app import app
from ..deps import get_audio_engine


@app.post("/api/audio/predict")
async def predict_audio(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No audio files uploaded.")

    uploads = []
    for file in files:
        content = await file.read()
        if not content:
            continue
        uploads.append((file.filename or "upload.wav", io.BytesIO(content)))

    if not uploads:
        raise HTTPException(status_code=400, detail="Uploaded audio files were empty.")

    try:
        engine = get_audio_engine()
        results = engine.predict_uploads(uploads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "results": [item.to_dict() for item in results],
        "meta": {
            "device": str(engine.device),
            "threshold": engine.config.threshold,
            "checkpoint": str(engine.config.checkpoint),
            "model": engine.config.model.name,
            "sample_rate": engine.config.sample_rate,
            "max_seconds": engine.config.max_seconds,
        },
    }
