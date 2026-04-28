from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json"
TRAINING_DIR = PROJECT_ROOT / "Training"
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
