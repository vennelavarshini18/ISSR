"""
evaluation metrics for audio quality
calculates scores to check how much speech improved after enhancement.

"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def evaluate(original_path: Path, enhanced_path: Path) -> dict:
    """calculates stoi, dnsmos, and si-sdr scores. returns nan if missing models."""
    return {
        "stoi": _stoi(original_path, enhanced_path),
        "dnsmos": _dnsmos(enhanced_path),
        "si_sdr": _si_sdr(original_path, enhanced_path),
    }


def _load(path: Path) -> tuple[np.ndarray, int]:
    """loads audio array, converts multi-channel to mono."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _stoi(ref_path: Path, enh_path: Path) -> float:
    """calculates short-time objective intelligibility score."""
    try:
        from pystoi import stoi
        ref, sr = _load(ref_path)
        enh, _ = _load(enh_path)

        n = min(len(ref), len(enh))
        return round(float(stoi(ref[:n], enh[:n], sr, extended=False)), 4)
    except Exception as e:
        logger.warning(f"stoi failed: {e}")
        return float("nan")


def _dnsmos(enh_path: Path) -> float:
    """returns nan placeholder until local dnsmos model is added."""
    logger.warning("dnsmos model not added yet -- returning nan")
    return float("nan")


def _si_sdr(ref_path: Path, enh_path: Path) -> float:
    """calculates scale-invariant signal-to-distortion ratio."""
    try:
        ref, _ = _load(ref_path)
        enh, _ = _load(enh_path)

        n = min(len(ref), len(enh))
        ref, enh = ref[:n] - ref[:n].mean(), enh[:n] - enh[:n].mean()

        dot = np.dot(ref, enh)
        s_target = (dot / (np.dot(ref, ref) + 1e-8)) * ref
        e_noise = enh - s_target

        return round(float(
            10 * np.log10((np.dot(s_target, s_target) + 1e-8) /
                          (np.dot(e_noise, e_noise) + 1e-8))
        ), 4)
    except Exception as e:
        logger.warning(f"si-sdr failed: {e}")
        return float("nan")
