import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import numpy as np

from tcamp.enhance.enhance import AudioEnhancer
from tcamp.diarization.diarize import DiarizationPipeline
from tcamp.diarization.utils import save_diarization_results, parse_rttm, calculate_der, save_rttm
from tcamp.transcription.transcribe import TranscriptionPipeline

logger = logging.getLogger(__name__)

def compute_rms(audio_array: np.ndarray) -> float:
    """Computes Root Mean Square energy of an audio array."""
    if len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_array**2)))

class TCAMPPipeline:
    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token or os.environ.get("HF_TOKEN")
        self.audio_enhancer = AudioEnhancer()
        self.diarization_pipeline = None
        self.transcription_pipeline = None
        
    def process(
        self,
        input_audio: str | Path,
        output_dir: str | Path,
        enhance_method: Literal["deepfilter", "noisereduce"] = "deepfilter",
        reference_rttm: Optional[str | Path] = None,
        phase: Literal["all", "enhance", "diarize", "transcribe"] = "all",
        num_speakers: Optional[int] = None,
        apply_normalization: bool = False,
        vad_threshold: Optional[float] = None,
        transcription_model: str = "large-v2",
        transcription_device: str = "cpu",
        transcription_compute: str = "int8"
    ) -> Dict[str, Any]:
        """
        Runs the full TCAMP Dual-Path pipeline.
        Step 1: Enhance the audio.
        Step 2: Diarize the RAW audio (to prevent Pyannote from missing speech).
        Step 3: Transcribe segments (dynamically switching between DFN and RAW based on RMS energy).
        """
        input_path = Path(input_audio)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_audio_path = out_dir / f"enhanced_{enhance_method}_{input_path.name}"
        diarization_json_path = out_dir / f"diarization_{enhance_method}_{input_path.stem}.json"
        diarization_rttm_path = out_dir / f"diarization_{enhance_method}_{input_path.stem}.rttm"
        transcript_path = out_dir / f"transcript_{input_path.stem}.txt"
        
        results = {
            "input_audio": str(input_path),
            "enhanced_audio": str(enhanced_audio_path),
            "enhancement_metrics": {},
            "diarization_results": str(diarization_json_path),
            "diarization_rttm": str(diarization_rttm_path),
            "transcript_file": str(transcript_path),
            "der": None,
            "der_breakdown": None,
            "transcript_segments": []
        }
        
        # 1. Enhancement (Runs on Input)
        if phase in ["all", "enhance", "transcribe"]:
            if not enhanced_audio_path.exists() or phase == "enhance":
                logger.info(f"Step 1: Enhancing audio using {enhance_method}...")
                metrics = self.audio_enhancer.enhance(
                    input_path=input_path,
                    output_path=enhanced_audio_path,
                    method=enhance_method,
                    target_sr=16000,
                    apply_normalization=apply_normalization
                )
                results["enhancement_metrics"] = metrics
            else:
                logger.info("Step 1: Enhanced audio already exists, skipping generation.")
        
        # 2. Diarization (Runs on RAW audio for accurate timestamps)
        if phase in ["all", "diarize", "transcribe"]:
            logger.info("Step 2: Running diarization on RAW audio...")
            if self.diarization_pipeline is None:
                if not self.auth_token:
                    raise ValueError("HF_TOKEN is missing. Cannot initialize diarization pipeline.")
                self.diarization_pipeline = DiarizationPipeline(auth_token=self.auth_token, vad_threshold=vad_threshold)
                
            segments = self.diarization_pipeline.process(str(input_path), num_speakers=num_speakers)
            save_diarization_results(segments, str(diarization_json_path))
            save_rttm(segments, str(diarization_rttm_path), uri=input_path.stem)
            logger.info(f"Diarization complete. Saved to {diarization_json_path.name}")
            
            # Evaluation
            if reference_rttm is None:
                potential_rttm = input_path.parent / f"{input_path.stem}_ground_truth.rttm"
                if potential_rttm.exists():
                    reference_rttm = str(potential_rttm)
            
            if reference_rttm and Path(reference_rttm).exists():
                logger.info(f"Evaluating DER against {Path(reference_rttm).name}...")
                try:
                    ref_segments = parse_rttm(str(reference_rttm))
                    der_stats = calculate_der(ref_segments, segments)
                    results["der"] = der_stats["der"]
                    results["der_breakdown"] = der_stats
                    logger.info(f"DER: {der_stats['der']:.2%} [Miss: {der_stats['miss']:.2%} | FA: {der_stats['false_alarm']:.2%}]")
                except ImportError:
                    pass

        # 3. Transcription (Dual-Path Logic)
        if phase in ["all", "transcribe"]:
            logger.info("Step 3: Transcription initialization...")
            if self.transcription_pipeline is None:
                self.transcription_pipeline = TranscriptionPipeline(
                    model_size=transcription_model, 
                    device=transcription_device, 
                    compute_type=transcription_compute
                )
            
            logger.info("Running Segment-Level Dual-Path Transcription...")
            import soundfile as sf
            raw_audio, sr = sf.read(input_path)
            enhanced_audio, _ = sf.read(enhanced_audio_path)
            
            transcript_log = []
            
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                speaker = segment["speaker"]
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                raw_seg = raw_audio[start_sample:end_sample]
                dfn_seg = enhanced_audio[start_sample:end_sample]
                
                raw_rms = compute_rms(raw_seg)
                dfn_rms = compute_rms(dfn_seg)
                
                # Source Selection Logic
                if raw_rms > 0 and (dfn_rms / raw_rms) < 0.10:
                    selected_seg = raw_seg
                    source = "RAW"
                else:
                    selected_seg = dfn_seg
                    source = "DFN"
                    
                # Skip segments shorter than 0.5 seconds to prevent Whisper hallucinations on silence
                segment_duration = end_time - start_time
                if segment_duration < 0.5:
                    continue

                # Transcribe numpy array
                trans_result = self.transcription_pipeline.transcribe_audio(selected_seg)
                text = " ".join([s["text"] for s in trans_result["segments"]]).strip()
                
                # Filter out known Whisper hallucinations
                hallucinations = [
                    "", "you", "you.", "Thank you.", "Thank you", 
                    "Bye.", "Bye", "Yeah.", "Okay.", "Oh.", "Hmm."
                ]
                
                # If text is empty, a known hallucination, or contains watermark links, skip it entirely
                if not text or text.strip() in hallucinations or "www." in text or "http" in text:
                    continue
                    
                log_line = f"[{start_time:05.1f} - {end_time:05.1f}] {speaker} ({source}): {text}"
                print(log_line)
                transcript_log.append(log_line)
                results["transcript_segments"].append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker,
                    "source": source,
                    "text": text
                })
                
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("\n".join(transcript_log))
            logger.info(f"Transcription complete! Saved to {transcript_path}")
                
        return results
