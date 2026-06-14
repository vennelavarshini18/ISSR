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
        reference_rttm: Optional[str | Path] = None,
        phase: Literal["all", "enhance", "diarize"] = "all",
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the full TCAMP pipeline: Enhancement -> Diarization -> Evaluation.
        """
        input_path = Path(input_audio)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_audio_path = out_dir / f"enhanced_{input_path.name}"
        diarization_json_path = out_dir / f"diarization_{input_path.stem}.json"
        diarization_rttm_path = out_dir / f"diarization_{input_path.stem}.rttm"
        
        results = {
            "input_audio": str(input_path),
            "enhanced_audio": str(enhanced_audio_path),
            "enhancement_metrics": {},
            "diarization_results": str(diarization_json_path),
            "diarization_rttm": str(diarization_rttm_path),
            "der": None,
            "der_breakdown": None
        }
        
        # 1. Enhancement
        if phase in ["all", "enhance"]:
            logger.info(f"Step 1: Enhancing audio using {enhance_method}...")
            metrics = enhance_audio(
                input_path=input_path,
                output_path=enhanced_audio_path,
                method=enhance_method,
                target_sr=16000
            )
            results["enhancement_metrics"] = metrics
        else:
            logger.info("Skipping Step 1 (Enhancement). Using input directly for diarization if needed.")
            if not enhanced_audio_path.exists():
                enhanced_audio_path = input_path
                results["enhanced_audio"] = str(input_path)

        # 2. Diarization
        if phase in ["all", "diarize"]:
            logger.info("Step 2: Running Diarization...")
            if self.diarization_pipeline is None:
                if not self.auth_token:
                    raise ValueError("HF_TOKEN is missing. Cannot initialize DiarizationPipeline.")
                self.diarization_pipeline = DiarizationPipeline(auth_token=self.auth_token)
                
            from tcamp.diarization.utils import save_rttm
            segments = self.diarization_pipeline.process(str(enhanced_audio_path), num_speakers=num_speakers)
            save_diarization_results(segments, str(diarization_json_path))
            save_rttm(segments, str(diarization_rttm_path), uri=input_path.stem)
            logger.info(f"Diarization complete. Saved to {diarization_json_path.name} and {diarization_rttm_path.name}")
            
            # 3. Evaluation (Optional, only if Diarization ran)
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
                        der_stats = calculate_der(ref_segments, segments)
                        results["der"] = der_stats["der"]
                        results["der_breakdown"] = {
                            "miss": der_stats["miss"],
                            "false_alarm": der_stats["false_alarm"],
                            "confusion": der_stats["confusion"]
                        }
                        logger.info(f"Diarization Error Rate (DER): {der_stats['der']:.2%} "
                                    f"[Miss: {der_stats['miss']:.2%} | "
                                    f"FA: {der_stats['false_alarm']:.2%} | "
                                    f"Confusion: {der_stats['confusion']:.2%}]")
                    except ImportError:
                        logger.error("pyannote.metrics not installed. Skipping DER calculation.")
                else:
                    logger.warning(f"Reference RTTM not found at {ref_path}. Skipping evaluation.")
        else:
            logger.info("Skipping Step 2 (Diarization).")
            results["diarization_results"] = None
                
        return results
