import os
import torch
from typing import Optional, Dict, Any, List
from pyannote.audio import Pipeline

class DiarizationPipeline:
    """
    Handles speaker diarization for audio files using Pyannote.
    """
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize the diarization pipeline.
        
        Args:
            auth_token: HuggingFace access token required for pyannote models.
        """
        if not auth_token:
            auth_token = os.environ.get("HF_TOKEN")
            if not auth_token:
                raise ValueError(
                    "A Hugging Face access token must be provided either as an argument "
                    "or set in the HF_TOKEN environment variable."
                )
                
        self.auth_token = auth_token
        print("Initializing Pyannote Speaker Diarization Pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.auth_token
        )
        
        # Use GPU if available
        if torch.cuda.is_available():
            print("CUDA available! Using GPU for diarization.")
            self.pipeline.to(torch.device("cuda"))
        else:
            print("Using CPU for diarization.")
        
    def process(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run diarization on the provided audio file.
        
        Args:
            audio_path: Path to the audio file.
            num_speakers: Optional parameter if the number of speakers is known.
            
        Returns:
            List of dictionaries containing speaker segments and timestamps.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Running diarization on {audio_path}...")
        
        # Run pipeline
        if num_speakers is not None:
            diarization = self.pipeline(audio_path, num_speakers=num_speakers)
        else:
            diarization = self.pipeline(audio_path)
            
        return self._format_output(diarization)
        
    def _format_output(self, diarization_result) -> List[Dict[str, Any]]:
        """Format pyannote Annotation output into a standard list structure."""
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        return segments
