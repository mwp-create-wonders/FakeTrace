import io
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .inference_engine import MARCInferenceEngine


WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="MARC Local Web App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_engine() -> MARCInferenceEngine:
    return MARCInferenceEngine(load_config())


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status():
    try:
        engine = get_engine()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "ready": True,
        "device": str(engine.device),
        "checkpoint": str(engine.config.checkpoint),
        "threshold": engine.config.threshold,
        "image_size": engine.config.image_size,
        "batch_size": engine.config.batch_size,
    }


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
        engine = get_engine()
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
