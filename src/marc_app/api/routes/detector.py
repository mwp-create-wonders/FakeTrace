import io

from fastapi import File, HTTPException, UploadFile

from ..app import app
from ..deps import get_detector_engine


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploads = []
    for file in files:
        content = await file.read()
        if not content:
            continue
        uploads.append((file.filename or "upload", io.BytesIO(content)))

    if not uploads:
        raise HTTPException(status_code=400, detail="Uploaded files were empty.")

    try:
        engine = get_detector_engine()
        results = engine.predict_uploads(uploads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "results": [item.to_dict() for item in results],
        "meta": {
            "device": str(engine.device),
            "threshold": engine.config.threshold,
            "checkpoint": str(engine.config.checkpoint),
        },
    }
