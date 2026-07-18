import os
import json
import logging
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

from tcamp.transcription.transcribe import TranscriptionPipeline
from tcamp.diarization.utils import load_diarization_results

def compute_rms(audio_array: np.ndarray) -> float:
    if len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_array**2)))

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("investigate")

def run_experiment(meeting_id: str, models: list):
    logger = setup_logger()
    
    # Define paths
    input_audio = Path(f"screening_notebooks/sample_input_and_output_files/{meeting_id}.wav")
    enhanced_audio = Path(f"observations/outputs/enhanced_deepfilter_{meeting_id}.wav")
    diarization_json = Path(f"observations/outputs/diarization_deepfilter_{meeting_id}.json")
    out_dir = Path("observations/hallucination_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not all([input_audio.exists(), enhanced_audio.exists(), diarization_json.exists()]):
        logger.error(f"Missing required files for {meeting_id}. Make sure you have run the full pipeline on it first!")
        return
        
    logger.info(f"Loading audio for {meeting_id}...")
    raw_audio, sr = sf.read(input_audio)
    dfn_audio, _ = sf.read(enhanced_audio)
    
    logger.info("Loading Pyannote diarization timestamps...")
    segments = load_diarization_results(str(diarization_json))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    for model_size in models:
        logger.info(f"\n==============================================")
        logger.info(f"Testing WhisperX Model: {model_size}")
        logger.info(f"==============================================")
        
        transcript_path = out_dir / f"transcript_{meeting_id}_{model_size}.txt"
        
        # Instantiate model
        transcriber = TranscriptionPipeline(
            model_size=model_size, 
            device=device, 
            compute_type=compute_type
        )
        
        transcript_log = []
        
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment["speaker"]
            
            # Minimum duration safeguard
            if (end_time - start_time) < 0.5:
                continue
                
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            raw_seg = raw_audio[start_sample:end_sample]
            dfn_seg = dfn_audio[start_sample:end_sample]
            
            raw_rms = compute_rms(raw_seg)
            dfn_rms = compute_rms(dfn_seg)
            
            # Dual-Path Logic
            if raw_rms > 0 and (dfn_rms / raw_rms) < 0.10:
                selected_seg = raw_seg
                source = "RAW"
            else:
                selected_seg = dfn_seg
                source = "DFN"
                
            try:
                # Force English decoding
                trans_result = transcriber.transcribe_audio(selected_seg, language="en")
                text_segments = trans_result.get("segments", [])
                text = " ".join([s.get("text", "") for s in text_segments]).strip()
            except IndexError:
                continue
            except Exception as e:
                logger.warning(f"Failed at [{start_time}-{end_time}]: {e}")
                continue
                
            if not text:
                continue
                
            log_line = f"[{start_time:05.1f} - {end_time:05.1f}] {speaker} ({source}): {text}"
            print(log_line)
            transcript_log.append(log_line)
            
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_log))
        logger.info(f"Finished {model_size}. Saved to {transcript_path.name}")
        
        # Clear GPU memory before next model loads
        del transcriber
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", type=str, default="EN2002a", help="Meeting ID to investigate (e.g. EN2002a)")
    parser.add_argument("--models", nargs="+", default=["large-v2", "large-v3-turbo", "medium.en", "base.en"], help="List of models to test")
    args = parser.parse_args()
    
    run_experiment(args.meeting, args.models)
