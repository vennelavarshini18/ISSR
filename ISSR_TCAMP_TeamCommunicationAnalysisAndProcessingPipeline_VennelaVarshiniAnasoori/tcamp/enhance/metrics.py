"""
tcamp/enhance/metrics.py

Audio quality evaluation metrics for Stage 1 enhancement assessment.

Metrics used
------------
STOI   : Short-Time Objective Intelligibility, measures speech intelligibility
         Range [0, 1]. Requires reference (original) audio.
         Relevant because intelligibility directly affects transcription accuracy
         in Stage 3 (WhisperX WER).

DNSMOS : DNS Mean Opinion Score (Microsoft P.835 model), reference-free MOS
         Range [1, 5]. No clean reference needed.
         Primary metric for real TRIP Lab data, no clean reference available.

SI-SDR : Scale-Invariant Signal-to-Distortion Ratio (dB)
         Higher = cleaner signal reconstruction.
         Used for cross-method comparison on AMI IHM (reference available).
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def evaluate(original_path: Path, enhanced_path: Path) -> dict:
    """
    Compute all three evaluation metrics.
    Returns NaN (not error) for any metric that cannot be computed —
    ensures pipeline does not crash on real lab data with no reference.
    """
    return {
        "stoi":   _stoi(original_path, enhanced_path),
        "dnsmos": _dnsmos(enhanced_path),
        "si_sdr": _si_sdr(original_path, enhanced_path),
    }


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _stoi(ref_path: Path, enh_path: Path) -> float:
    try:
        from pystoi import stoi
        ref, sr = _load(ref_path)
        enh, _  = _load(enh_path)
        n = min(len(ref), len(enh))
        return round(float(stoi(ref[:n], enh[:n], sr, extended=False)), 4)
    except Exception as e:
        logger.warning(f"STOI failed: {e}")
        return float("nan")


def _dnsmos(enh_path: Path) -> float:
    """
    DNSMOS P.835 via ONNX inference (Microsoft DNS Challenge model).
    Reference-free — primary metric for unlabelled TRIP Lab recordings.
    """
    try:
        # Wire in ONNX DNSMOS model here
        # See: https://github.com/microsoft/DNS-Challenge
        raise NotImplementedError(
            "DNSMOS ONNX model integration pending. "
            "Download model from DNS-Challenge repo and wire into _dnsmos()."
        )
    except NotImplementedError:
        logger.warning("DNSMOS not yet integrated — returning NaN")
        return float("nan")
    except Exception as e:
        logger.warning(f"DNSMOS failed: {e}")
        return float("nan")


def _si_sdr(ref_path: Path, enh_path: Path) -> float:
    try:
        ref, _ = _load(ref_path)
        enh, _ = _load(enh_path)
        n = min(len(ref), len(enh))
        ref, enh = ref[:n] - ref[:n].mean(), enh[:n] - enh[:n].mean()
        dot = np.dot(ref, enh)
        s_target = (dot / (np.dot(ref, ref) + 1e-8)) * ref
        e_noise  = enh - s_target
        return round(float(
            10 * np.log10((np.dot(s_target, s_target) + 1e-8) /
                          (np.dot(e_noise,  e_noise)  + 1e-8))
        ), 4)
    except Exception as e:
        logger.warning(f"SI-SDR failed: {e}")
        return float("nan")
