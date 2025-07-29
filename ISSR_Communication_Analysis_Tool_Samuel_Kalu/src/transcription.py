# ===================================== ENFORCE CACHE DIRECTORY======================================
import os

os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"
# ====================================================================================================

import warnings

import torch
from faster_whisper import WhisperModel


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper, returning segments with timestamps.
    Args:
        audio_path (str): Path to the audio file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use medium.en model with faster-whisper
    model_size = "medium.en"
    model = WhisperModel(
        model_size,
        device=device,
        compute_type="int8",
        download_root="./model_weights/whisper",
    )
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    # Convert segments generator to list of dicts for compatibility
    texts = [
        segment.text
        for segment in segments
        if hasattr(segment, "text") and segment.text
    ]
    transcription = " ".join(texts)
    return transcription
