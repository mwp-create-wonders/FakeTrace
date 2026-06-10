from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json"
MODEL_HUB_DIR = PROJECT_ROOT / "models"
MARC_MODEL_DIR = MODEL_HUB_DIR / "marc"
TRUFOR_MODEL_DIR = MODEL_HUB_DIR / "trufor"
CATNET_MODEL_DIR = MODEL_HUB_DIR / "CAT-Net"
FASSA_MODEL_DIR = MODEL_HUB_DIR / "Fassa"
EFFUNETPP_MODEL_DIR = MODEL_HUB_DIR / "effunetpp"
TRI_MODEL_DIR = MODEL_HUB_DIR / "TRI"
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
