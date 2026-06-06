import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal

from tcamp.enhance.enhance import enhance_audio
from tcamp.diarization.diarize import DiarizationPipeline
from tcamp.diarization.utils import save_diarization_results, parse_rttm, calculate_der

logger = logging.getLogger(__name__)

class TCAMPPipeline:
    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token or os.environ.get("HF_TOKEN")
        self.diarization_pipeline = None
        
    def process(
        self,
        input_audio: str | Path,
        output_dir: str | Path,
        enhance_method: Literal["deepfilter", "noisereduce"] = "deepfilter",
        reference_rttm: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Runs the full TCAMP pipeline: Enhancement -> Diarization -> Evaluation.
        """
        input_path = Path(input_audio)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_audio_path = out_dir / f"enhanced_{input_path.name}"
        diarization_json_path = out_dir / f"diarization_{input_path.stem}.json"
        
        results = {
            "input_audio": str(input_path),
            "enhanced_audio": str(enhanced_audio_path),
            "enhancement_metrics": {},
            "diarization_results": str(diarization_json_path),
            "der": None
        }
        
        # 1. Enhancement
        logger.info(f"Step 1: Enhancing audio using {enhance_method}...")
        metrics = enhance_audio(
            input_path=input_path,
            output_path=enhanced_audio_path,
            method=enhance_method,
            target_sr=16000
        )
        results["enhancement_metrics"] = metrics
        
        # 2. Diarization
        logger.info("Step 2: Running Diarization...")
        if self.diarization_pipeline is None:
            if not self.auth_token:
                raise ValueError("HF_TOKEN is missing. Cannot initialize DiarizationPipeline.")
            self.diarization_pipeline = DiarizationPipeline(auth_token=self.auth_token)
            
        segments = self.diarization_pipeline.process(str(enhanced_audio_path))
        save_diarization_results(segments, str(diarization_json_path))
        logger.info(f"Diarization complete. Saved to {diarization_json_path.name}")
        
        # 3. Evaluation (Optional)
        if reference_rttm is None:
            potential_rttm = input_path.parent / f"{input_path.stem}_ground_truth.rttm"
            obs_rttm = Path("observations") / f"{input_path.stem}_ground_truth.rttm"
            if potential_rttm.exists():
                reference_rttm = str(potential_rttm)
            elif obs_rttm.exists():
                reference_rttm = str(obs_rttm)

        if reference_rttm:
            ref_path = Path(reference_rttm)
            if ref_path.exists():
                logger.info(f"Step 3: Evaluating against ground truth {ref_path.name}...")
                ref_segments = parse_rttm(str(ref_path))
                try:
                    der = calculate_der(ref_segments, segments)
                    results["der"] = der
                    logger.info(f"Diarization Error Rate (DER): {der:.2%}")
                except ImportError:
                    logger.error("pyannote.metrics not installed. Skipping DER calculation.")
            else:
                logger.warning(f"Reference RTTM not found at {ref_path}. Skipping evaluation.")
                
        return results
