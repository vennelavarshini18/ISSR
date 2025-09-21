# ===================================== ENFORCE CACHE DIRECTORY======================================
import os

os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"
# ====================================================================================================

import warnings

import torch
from faster_whisper import WhisperModel
from loguru import logger


def transcribe_audio(audio_path):
    """
    Transcribe audio using Whisper, with proper handling for short/invalid audio files.
    Returns empty string if transcription fails or audio is invalid.
    
    Args:
        audio_path (str): Path to the audio file.
    
    Returns:
        str: Transcribed text or empty string if failed
    """
    
    #  Set up model with proper error handling
    logger.debug(f"Starting transcription for {audio_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "medium.en"
    
    # Initialize transcription result to empty string (safe default)
    transcription = ""
    
    try:
        # Load model
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",
            download_root="./model_weights/whisper",
        )
        
        # Transcribe
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        
        # Convert segments to text
        texts = [
            segment.text
            for segment in segments
            if hasattr(segment, "text") and segment.text
        ]
        
        # Join texts or return empty string if none
        transcription = " ".join(texts) if texts else ""
        
        # Log if we got empty result
        if not transcription:
            logger.warning(f"Empty transcription result for {audio_path}")
            
    except Exception as e:
        logger.error(f"Whisper transcription failed for {audio_path}: {str(e)}")
        # Keep transcription as empty string

    return transcription