from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse

from ..app import app
from ...features.detector_report.service import build_detector_report, resolve_detector_report_path


@app.post("/api/predict/report")
@app.post("/api/detector/report")
async def create_detector_report(request: Request):
    try:
        payload = await request.json()
        generated = build_detector_report(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "report_id": generated.report_id,
        "pdf_url": f"/api/predict/report/{generated.report_id}",
        "download_url": f"/api/predict/report/{generated.report_id}?download=1",
    }


@app.get("/api/predict/report/{report_id}")
@app.get("/api/detector/report/{report_id}")
def get_detector_report(report_id: str, download: bool = Query(False)):
    try:
        path = resolve_detector_report_path(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    disposition = "attachment" if download else "inline"
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'{disposition}; filename="{path.name}"'},
    )
