from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..ui.assets import STATIC_DIR


app = FastAPI(title="FakeTrace Web App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Import routes after app creation so each module can register its endpoints.
<<<<<<< Updated upstream
from .routes import audio, detector, localization, pages, system, video  # noqa: F401,E402
=======
from .routes import detector, pages, security, system, trufor  # noqa: F401,E402
>>>>>>> Stashed changes
