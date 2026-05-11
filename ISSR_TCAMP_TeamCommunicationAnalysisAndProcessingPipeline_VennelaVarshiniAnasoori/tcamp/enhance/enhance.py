"""
tcamp/enhance/enhance.py

Stage 1: Audio Enhancement
==========================
Primary method : DeepFilterNet3 (DL-based, full-band 48kHz)
Baseline       : NoiseReduce (spectral gating, reference-free)

Design rationale
----------------
TRIP Lab headset recordings present three noise profiles:
  1. Low-frequency equipment hum (50–60 Hz)
  2. Push-to-talk click artifacts
  3. Variable room reverberation

DeepFilterNet3 is selected as primary because:
  - Full-band 48kHz processing (vs 16kHz for most speech enhancement models)
  - RTF ~0.19 on CPU, practical for long simulation session recordings
  - Perceptual loss training: optimizes for human intelligibility, not just SNR

NoiseReduce is retained as classical baseline because:
  - Reference-free (no clean audio required — matches real lab conditions)
  - Led on DNSMOS (1.725) in screening task evaluation
  - Serves as fallback if DeepFilterNet3 unavailable

Evaluation metrics
------------------
  STOI    : Short-Time Objective Intelligibility [0,1], higher = more intelligible
  DNSMOS  : Reference-free MOS prediction (Microsoft P.835), higher = better quality
  SI-SDR  : Scale-Invariant SDR in dB, higher = cleaner signal
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def enhance_audio(
    input_path: str | Path,
    output_path: str | Path,
    method: Literal["deepfilter", "noisereduce"] = "deepfilter",
) -> dict:
    """
    Enhance raw headset audio. Main entry point for Stage 1.

    Args:
        input_path  : Path to raw .WAV (mono or multi-channel)
        output_path : Path to write enhanced .WAV
        method      : 'deepfilter' (primary DL) | 'noisereduce' (classical baseline)

    Returns:
        dict — {method, stoi, dnsmos, si_sdr}

    Raises:
        FileNotFoundError : input_path does not exist
        ValueError        : unsupported method string
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input audio not found: {input_path}\n"
            "Verify the path and confirm the file is a .WAV format."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Stage 1 — Audio Enhancement] Method: {method} | Input: {input_path.name}")

    if method == "deepfilter":
        _run_deepfilter(input_path, output_path)
    elif method == "noisereduce":
        from .baseline import run_noisereduce
        run_noisereduce(input_path, output_path)
    else:
        raise ValueError(
            f"Unsupported method: '{method}'. Choose 'deepfilter' or 'noisereduce'."
        )

    from .metrics import evaluate
    results = evaluate(original_path=input_path, enhanced_path=output_path)
    results["method"] = method

    logger.info(f"[Stage 1 — Complete] {results}")
    return results


def _run_deepfilter(input_path: Path, output_path: Path) -> None:
    """
    Run DeepFilterNet3 enhancement.

    Processes audio at native 48kHz (resampling applied automatically).
    No fine-tuning required — pretrained weights generalize well to
    headset microphone noise profiles per Schröter et al. (2023).
    """
    try:
        from df.enhance import enhance, init_df
        from df.io import load_audio, save_audio
    except ImportError:
        raise ImportError(
            "DeepFilterNet not installed.\n"
            "Install with: pip install deepfilternet"
        )

    logger.info("[DeepFilterNet3] Loading model...")
    model, df_state, _ = init_df()

    audio, _ = load_audio(str(input_path), sr=df_state.sr())
    logger.info(f"[DeepFilterNet3] Enhancing at {df_state.sr()} Hz...")
    enhanced = enhance(model, df_state, audio)

    save_audio(str(output_path), enhanced, df_state.sr())
    logger.info(f"[DeepFilterNet3] Saved to {output_path.name}")
