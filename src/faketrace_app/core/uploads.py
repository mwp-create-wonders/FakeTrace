import re
from pathlib import Path


_SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_upload_filename(filename: str | None, default: str = "upload") -> str:
    if not filename:
        return default

    # Browsers and desktop clients may send Windows-style paths such as
    # C:\fakepath\image.png even when the server runs on Linux.
    normalized = filename.replace("\\", "/").strip()
    basename = Path(normalized).name.strip()
    return basename or default


def safe_upload_stem(filename: str | None, default: str = "upload") -> str:
    stem = Path(normalize_upload_filename(filename, default=default)).stem.strip() or default
    sanitized = _SAFE_STEM_PATTERN.sub("_", stem).strip("._")
    return sanitized or default
