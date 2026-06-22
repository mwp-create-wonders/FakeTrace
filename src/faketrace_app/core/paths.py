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
AUDIO_MODEL_DIR = MODEL_HUB_DIR / "audio"
FORENSIC_MOE_MODEL_DIR = MODEL_HUB_DIR / "Forensic-MoE-main"
FORGELENS_MODEL_DIR = MODEL_HUB_DIR / "ForgeLens"
LOTA_MODEL_DIR = MODEL_HUB_DIR / "LOTA"
MF2DA_MODEL_DIR = MODEL_HUB_DIR / "MF2DA"
UNIVFD_MODEL_DIR = MODEL_HUB_DIR / "UnivFD"
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VALID_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
VALID_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
