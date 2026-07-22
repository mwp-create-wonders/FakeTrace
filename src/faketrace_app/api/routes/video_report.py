from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse

from ..app import app
from ...features.video_report.service import build_video_report, resolve_video_report_path
from ...features.video_report.task_store import create_video_task


def _convert_tri_result_to_report_item(result) -> dict:
    fake_probability = float(getattr(result, "probability", 0.5))
    prediction_int = getattr(result, "prediction", 0)

    if prediction_int == 1:
        prediction = "fake"
    elif prediction_int == 0:
        prediction = "real"
    else:
        prediction = "error"

    return {
        "filename": getattr(result, "filename", ""),
        "duration": float(getattr(result, "duration", 0.0)),
        "width": int(getattr(result, "width", 0)),
        "height": int(getattr(result, "height", 0)),
        "fps": float(getattr(result, "fps", 0.0)),
        "total_frames": int(getattr(result, "total_frames", 0)),
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": float(1.0 - fake_probability),
        "threshold": float(getattr(result, "threshold", 0.5)),
        "frame_info": getattr(result, "frame_info", None),
        "velocity_l2": getattr(result, "velocity_l2", None),
        "acceleration_l2": getattr(result, "acceleration_l2", None),
        "lota_scores": getattr(result, "lota_scores", None),
        "suspicious_frame_b64": getattr(result, "suspicious_frame_b64", None),
        "suspicious_frame_time": getattr(result, "suspicious_frame_time", None),
    }


@app.post("/api/video/report")
async def create_video_report(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    model_key = str(payload.get("model") or "tri").lower()
    model_name = payload.get("model_name") or "TRI"
    include_ai_analysis = bool(payload.get("include_ai_analysis"))

    results = payload.get("items") or []
    items = []
    for result in results:
        if hasattr(result, "filename"):
            items.append(_convert_tri_result_to_report_item(result))
        elif isinstance(result, dict):
            items.append(result)

    video_count = len(items)
    if video_count == 0:
        raise HTTPException(status_code=400, detail="No video results provided")

    task = create_video_task(model=model_key, video_count=video_count)

    report_payload = {
        "model": model_key,
        "model_name": model_name,
        "test_id": task.test_id,
        "include_ai_analysis": include_ai_analysis,
        "upload_time": task.created_at,
        "items": items,
    }

    try:
        generated = build_video_report(report_payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "report_id": generated.report_id,
        "test_id": task.test_id,
        "pdf_url": f"/api/video/report/{generated.report_id}",
        "download_url": f"/api/video/report/{generated.report_id}?download=1",
    }


@app.get("/api/video/report/{report_id}")
def get_video_report(report_id: str, download: bool = Query(False)):
    try:
        path = resolve_video_report_path(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    disposition = "attachment" if download else "inline"
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'{disposition}; filename="{path.name}"'},
    )
