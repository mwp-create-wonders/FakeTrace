import io
from typing import Literal

from fastapi import File, HTTPException, Query, UploadFile

from ..app import app
from ..deps import (
    get_detector_engine,
    get_forensic_moe_engine,
    get_forgelens_engine,
    get_lota_engine,
    get_mf2da_engine,
    get_univfd_engine,
)


@app.post("/api/predict")
async def predict(
    files: list[UploadFile] = File(...),
    model: Literal["marc", "forensic_moe", "forgelens", "lota", "mf2da", "univfd"] = Query(
        "marc",
        description="Image detection model to use",
    ),
):
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
        if model == "marc":
            engine = get_detector_engine()
            engine_name = "MARC"
        elif model == "forensic_moe":
            engine = get_forensic_moe_engine()
            engine_name = "Forensic-MoE"
        elif model == "forgelens":
            engine = get_forgelens_engine()
            engine_name = "ForgeLens"
        elif model == "lota":
            engine = get_lota_engine()
            engine_name = "LOTA"
        elif model == "mf2da":
            engine = get_mf2da_engine()
            engine_name = "MF2DA"
        elif model == "univfd":
            engine = get_univfd_engine()
            engine_name = "UnivFD"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown image detector: {model}")
        results = engine.predict_uploads(uploads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    device = str(getattr(engine, "device", "unknown"))
    threshold = getattr(getattr(engine, "config", engine), "threshold", getattr(engine, "threshold", 0.5))

    return {
        "results": [item.to_dict() for item in results],
        "meta": {
            "device": device,
            "threshold": threshold,
            "checkpoint": str(getattr(getattr(engine, "config", engine), "checkpoint", "")),
            "model": engine_name,
        },
    }
