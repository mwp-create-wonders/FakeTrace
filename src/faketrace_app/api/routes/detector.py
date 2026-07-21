import io
import traceback
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
    reset_detector_engine_cache,
)
from ...features.detector_report.task_store import create_detector_task


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    cuda_markers = [
        "out of memory",
        "cuda",
        "cublas",
        "cudnn",
        "device-side assert",
    ]
    return any(marker in message for marker in cuda_markers)


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
        traceback.print_exc()
        if model != "marc" or not _is_cuda_runtime_error(exc):
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            reset_detector_engine_cache()
            engine = get_detector_engine("cpu")
            engine_name = "MARC"
            rewound_uploads = []
            for filename, file_obj in uploads:
                file_obj.seek(0)
                rewound_uploads.append((filename, file_obj))
            results = engine.predict_uploads(rewound_uploads)
        except Exception as cpu_exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"GPU 推理失败，切换 CPU 重试后仍失败: {cpu_exc}",
            ) from cpu_exc

    device = str(getattr(engine, "device", "unknown"))
    threshold = getattr(getattr(engine, "config", engine), "threshold", getattr(engine, "threshold", 0.5))
    task = create_detector_task(model=model, image_count=len(results))

    return {
        "results": [item.to_dict() for item in results],
        "meta": {
            "device": device,
            "threshold": threshold,
            "checkpoint": str(getattr(getattr(engine, "config", engine), "checkpoint", "")),
            "model": engine_name,
        },
        "detector_task_id": task.id,
        "detector_test_id": task.test_id,
    }
