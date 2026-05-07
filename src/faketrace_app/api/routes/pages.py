from fastapi.responses import FileResponse, HTMLResponse

from ...ui.assets import UI_DIR
from ..app import app


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(UI_DIR / "index.html")
