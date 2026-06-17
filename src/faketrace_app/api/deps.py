from functools import lru_cache

from ..core.config import load_config


@lru_cache(maxsize=1)
def get_detector_engine():
    from ..features.detector.service import MARCInferenceEngine

    return MARCInferenceEngine(load_config())


@lru_cache(maxsize=1)
def get_audio_engine():
    from ..features.audio.service import AudioDeepfakeEngine

    return AudioDeepfakeEngine(load_config())


@lru_cache(maxsize=1)
def get_trufor_engine():
    from ..features.trufor.service import TruForLocalizationEngine

    return TruForLocalizationEngine()


@lru_cache(maxsize=1)
def get_catnet_engine():
    from ..features.catnet.service import CATNetLocalizationEngine

    return CATNetLocalizationEngine()


@lru_cache(maxsize=1)
def get_fassa_engine():
    from ..features.fassa.service import FassaLocalizationEngine

    return FassaLocalizationEngine()


@lru_cache(maxsize=1)
def get_effunetpp_engine():
    from ..features.effunetpp.service import EffunetPPLocalizationEngine

    return EffunetPPLocalizationEngine()
