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
) -> dict:
    """
    cleans up audio using the chosen method.

    args:
        input_path: path to the raw .wav file
        output_path: path to save the cleaned .wav file
        method: 'deepfilter' or 'noisereduce'

    returns:
        dict with quality scores and method name
    """
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        raise FileNotFoundError(f"input file not found: {in_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"running {method} on {in_file.name}")

    if method == "deepfilter":
        _run_deepfilter(in_file, out_file)
    elif method == "noisereduce":
        from .baseline import run_noisereduce
        run_noisereduce(in_file, out_file)
    else:
        raise ValueError(f"unknown method '{method}'. choose 'deepfilter' or 'noisereduce'.")

    from .metrics import evaluate
    scores = evaluate(original_path=in_file, enhanced_path=out_file)
    scores["method"] = method

    logger.info(f"finished. scores: {scores}")
    return scores


def _run_deepfilter(in_file: Path, out_file: Path) -> None:
    """runs deepfilternet3 enhancement at native 48kHz."""
    try:
        from df.enhance import enhance, init_df
        from df.io import load_audio, save_audio
    except ImportError:
        raise ImportError("deepfilternet not installed. run: pip install deepfilternet")

    logger.info("loading deepfilternet model...")
    model, state, _ = init_df()

    audio, _ = load_audio(str(in_file), sr=state.sr())
    enhanced = enhance(model, state, audio)

    save_audio(str(out_file), enhanced, state.sr())
    logger.info(f"saved deepfilter output to {out_file.name}")
