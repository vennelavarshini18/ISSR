"""
tcamp/enhance/baseline.py

Classical audio enhancement baselines.
Kept separate from the primary DL method for clean ablation study comparisons.

Baseline methods evaluated in screening task (3-min AMI IHM sample):
    Spectral Subtraction : DNSMOS 1.421 | STOI 0.610
    Wiener Filter        : DNSMOS 1.288 | STOI 0.638
    NoiseReduce          : DNSMOS 1.725 | STOI 0.633  <- selected baseline
    Combined Pipeline    : DNSMOS 1.275 | SNR -6.87 dB (over-processing artifact)

Finding: Single-method selection outperforms naive chaining. NoiseReduce
retained as baseline -- only reference-free metric (DNSMOS) led, making it
most suitable for real lab data with no clean reference available.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _load_mono(path: Path, target_sr: int = 16000):
    """Load audio, convert to mono, resample if needed."""
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio, sr


def run_noisereduce(input_path: Path, output_path: Path) -> None:
    """
    NoiseReduce (spectral gating) baseline enhancement.

    Selected over Wiener and spectral subtraction because DNSMOS (1.725)
    led all classical methods -- only reference-free metric, most valid
    for real lab recordings with no clean reference audio.
    """
    try:
        import noisereduce as nr
    except ImportError:
        raise ImportError("Run: pip install noisereduce")

    audio, sr = _load_mono(input_path)
    logger.info("[NoiseReduce] Applying spectral gating...")
    reduced = nr.reduce_noise(y=audio, sr=sr)
    sf.write(str(output_path), reduced, sr)
    logger.info(f"[NoiseReduce] Saved to {output_path.name}")


def run_wiener(input_path: Path, output_path: Path) -> None:
    """
    Wiener filter baseline. Retained for ablation -- not selected as primary
    baseline (DNSMOS 1.288, below NoiseReduce).
    """
    from scipy.signal import wiener
    audio, sr = _load_mono(input_path)
    filtered = wiener(audio)
    sf.write(str(output_path), filtered.astype(np.float32), sr)
