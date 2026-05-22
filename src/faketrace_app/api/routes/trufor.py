import io
from pathlib import Path
from typing import Literal

from fastapi import File, HTTPException, UploadFile, Query

from ..app import app
from ..deps import get_trufor_engine, get_catnet_engine, get_fassa_engine, get_effunetpp_engine


@app.post("/api/localize")
async def localize(
    files: list[UploadFile] = File(...),
    model: Literal["trufor", "catnet", "fassa", "effunetpp"] = Query("trufor", description="Model to use for localization"),
    save: bool = Query(False, description="Whether to save results to disk"),
    output_dir: str = Query("output", description="Directory to save output files"),
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
        if model == "trufor":
            engine = get_trufor_engine()
            results = engine.predict_uploads(uploads, save=save, output_dir=Path(output_dir))
            meta = {
                "device": str(engine.device),
                "model": "TruFor",
                "experiment": engine.config.experiment,
                "model_file": str(engine.config.model_file),
            }
        elif model == "catnet":
            engine = get_catnet_engine()
            results = engine.predict_uploads(uploads, save=save, output_dir=Path(output_dir))
            meta = {
                "device": str(engine.device),
                "model": "CAT-Net",
                "model_file": str(engine.config.TEST.MODEL_FILE),
            }
        elif model == "fassa":
            engine = get_fassa_engine()
            results = engine.predict_uploads(uploads, save=save, output_dir=Path(output_dir))
            meta = {
                "device": str(engine.device),
                "model": "Fassa",
                "model_file": str((engine.FASSA_MODEL_DIR / "fassa_best_model.pth").resolve()),
            }
        elif model == "effunetpp":
            engine = get_effunetpp_engine()
            results = engine.predict_uploads(uploads, save=save, output_dir=Path(output_dir))
            meta = {
                "device": str(engine.device),
                "model": "EffUnetPP",
                "model_file": str((engine.EFFUNETPP_MODEL_DIR / "effunetpp_best_model.pth").resolve()),
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "results": [item.to_dict() for item in results],
        "meta": meta,
    }
