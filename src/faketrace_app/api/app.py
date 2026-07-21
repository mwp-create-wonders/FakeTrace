from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..ui.assets import STATIC_DIR


app = FastAPI(title="FakeTrace Web App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Import routes after app creation so each module can register its endpoints.
from .routes import (  # noqa: F401,E402
    audio,
    audio_report,
    detector,
    detector_report,
    localization,
    pages,
    security,
    system,
    video,
)
