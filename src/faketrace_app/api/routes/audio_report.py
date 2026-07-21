from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse

from ..app import app
from ...features.audio_report.service import build_audio_report, resolve_audio_report_path


@app.post("/api/audio/report")
async def create_audio_report(request: Request):
    try:
        payload = await request.json()
        generated = build_audio_report(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "report_id": generated.report_id,
        "pdf_url": f"/api/audio/report/{generated.report_id}",
        "download_url": f"/api/audio/report/{generated.report_id}?download=1",
    }


@app.get("/api/audio/report/{report_id}")
def get_audio_report(report_id: str, download: bool = Query(False)):
    try:
        path = resolve_audio_report_path(report_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    disposition = "attachment" if download else "inline"
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'{disposition}; filename="{path.name}"'},
    )
