from functools import lru_cache

from ..core.config import load_config
from ..features.catnet.service import CATNetLocalizationEngine
from ..features.detector.service import MARCInferenceEngine
from ..features.fassa.service import FassaLocalizationEngine
from ..features.trufor.service import TruForLocalizationEngine


@lru_cache(maxsize=1)
def get_detector_engine() -> MARCInferenceEngine:
    return MARCInferenceEngine(load_config())


@lru_cache(maxsize=1)
def get_trufor_engine() -> TruForLocalizationEngine:
    return TruForLocalizationEngine()


@lru_cache(maxsize=1)
def get_catnet_engine() -> CATNetLocalizationEngine:
    return CATNetLocalizationEngine()


@lru_cache(maxsize=1)
def get_fassa_engine() -> FassaLocalizationEngine:
    return FassaLocalizationEngine()
