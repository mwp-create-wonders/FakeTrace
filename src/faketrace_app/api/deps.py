from __future__ import annotations

from functools import lru_cache

from ..core.config import load_config
from ..features.detector.service import MARCInferenceEngine


@lru_cache(maxsize=2)
def get_detector_engine(device: str | None = None) -> MARCInferenceEngine:
    overrides = {"device": device} if device else None
    return MARCInferenceEngine(load_config(overrides=overrides))


def reset_detector_engine_cache() -> None:
    get_detector_engine.cache_clear()


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


@lru_cache(maxsize=1)
def get_forensic_moe_engine():
    from ..features.detector.multi_service import ForensicMoEDetector

    return ForensicMoEDetector(device=load_config().device)


@lru_cache(maxsize=1)
def get_forgelens_engine():
    from ..features.detector.multi_service import ForgeLensDetector

    return ForgeLensDetector(device=load_config().device)


@lru_cache(maxsize=1)
def get_lota_engine():
    from ..features.detector.multi_service import LOTADetector

    return LOTADetector(device=load_config().device)


@lru_cache(maxsize=1)
def get_mf2da_engine():
    from ..features.detector.multi_service import MF2DADetector

    return MF2DADetector(device=load_config().device)


@lru_cache(maxsize=1)
def get_univfd_engine():
    from ..features.detector.multi_service import UnivFDDetector

    return UnivFDDetector(device=load_config().device)


@lru_cache(maxsize=1)
def get_tri_engine():
    from ..features.TRI.service import TRIDetectionEngine

    return TRIDetectionEngine(device=load_config().device)
