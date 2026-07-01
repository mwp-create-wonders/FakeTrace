from fastapi import HTTPException

from ...features.security.service import SecurityScanService
from ..app import app


@app.post("/api/security/scan")
def run_security_scan():
    try:
        service = SecurityScanService()
        return service.run_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
