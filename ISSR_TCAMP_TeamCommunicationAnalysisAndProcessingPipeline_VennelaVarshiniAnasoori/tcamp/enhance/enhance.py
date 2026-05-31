"""
audio enhancement core interface
removes background hum and simulator noise from headset recordings.

"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def enhance_audio(
    input_path: str | Path,
    output_path: str | Path,
    method: Literal["deepfilter", "noisereduce"] = "deepfilter",
    target_sr: int = 16000,
) -> dict:
    """
    cleans up audio using the chosen method.

    args:
        input_path: path to the raw .wav file
        output_path: path to save the cleaned .wav file
        method: 'deepfilter' or 'noisereduce'
        target_sr: sampling rate to enforce across the pipeline (default 16000)

    returns:
        dict with quality scores and method name
    """
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        raise FileNotFoundError(f"input file not found: {in_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"running {method} on {in_file.name} (target {target_sr} hz)")

    if method == "deepfilter":
        _run_deepfilter(in_file, out_file, target_sr)
    elif method == "noisereduce":
        from .baseline import run_noisereduce
        run_noisereduce(in_file, out_file, target_sr)
    else:
        raise ValueError(f"unknown method '{method}'. choose 'deepfilter' or 'noisereduce'.")

    from .metrics import evaluate
    scores = evaluate(original_path=in_file, enhanced_path=out_file)
    scores["method"] = method

    logger.info(f"finished. scores: {scores}")
    return scores


def _run_deepfilter(in_file: Path, out_file: Path, target_sr: int) -> None:
    """runs deepfilternet3 enhancement and aligns output to target_sr."""
    try:
        from df.enhance import enhance, init_df
        from df.io import load_audio, save_audio
        import torchaudio.functional as F
    except ImportError:
        raise ImportError("deepfilternet or torchaudio not installed. check requirements.")

    logger.info("loading deepfilternet model...")
    model, state, _ = init_df()

    audio, _ = load_audio(str(in_file), sr=state.sr())
    enhanced = enhance(model, state, audio)

    # standardize back to target frequency before saving
    if state.sr() != target_sr:
        logger.info(f"resampling deepfilter output from {state.sr()} to {target_sr} hz")
        enhanced = F.resample(enhanced, state.sr(), target_sr)

    save_audio(str(out_file), enhanced, target_sr)
    logger.info(f"saved deepfilter output to {out_file.name} at {target_sr} hz")
