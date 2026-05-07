from functools import lru_cache

from ..core.config import load_config
from ..features.detector.service import MARCInferenceEngine
from ..features.trufor.service import TruForLocalizationEngine


@lru_cache(maxsize=1)
def get_detector_engine() -> MARCInferenceEngine:
    return MARCInferenceEngine(load_config())


@lru_cache(maxsize=1)
def get_trufor_engine() -> TruForLocalizationEngine:
    return TruForLocalizationEngine()
