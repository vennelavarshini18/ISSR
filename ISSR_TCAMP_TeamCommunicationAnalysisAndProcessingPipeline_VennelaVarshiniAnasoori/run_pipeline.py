import argparse
import logging
import json
from tcamp.pipeline import TCAMPPipeline

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="TCAMP Audio Processing Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output-dir", "-o", type=str, default="observations/outputs", help="Directory to save output files")
    parser.add_argument("--enhance-method", "-m", type=str, choices=["deepfilter", "noisereduce"], default="deepfilter", help="Enhancement method")
    parser.add_argument("--evaluate", "-e", type=str, help="Path to reference RTTM file for DER evaluation")
    parser.add_argument("--phase", type=str, choices=["all", "enhance", "diarize"], default="all", help="Which pipeline phase to run")
    parser.add_argument("--num-speakers", "-s", type=int, help="Force a specific number of speakers (optional)")
    parser.add_argument("--vad-threshold", type=float, help="Override Pyannote VAD segmentation threshold (e.g., 0.30)")
    parser.add_argument("--apply-normalization", action="store_true", help="Apply DRC speech normalization after enhancement")
    parser.add_argument("--token", "-t", type=str, help="Hugging Face token (defaults to HF_TOKEN env var)")
    
    args = parser.parse_args()
    setup_logger()
    logger = logging.getLogger("run_pipeline")
    
    logger.info("initializing tcamp pipeline...")
    try:
        pipeline = TCAMPPipeline(auth_token=args.token)
        results = pipeline.process(
            input_audio=args.input,
            output_dir=args.output_dir,
            enhance_method=args.enhance_method,
            reference_rttm=args.evaluate,
            phase=args.phase,
            num_speakers=args.num_speakers,
            apply_normalization=args.apply_normalization,
            vad_threshold=args.vad_threshold
        )
        
        logger.info("\npipeline execution summary:")
        print(json.dumps(results, indent=2))
        
    except ValueError as ve:
        logger.error(f"config error: {str(ve)}")
    except Exception as ex:
        logger.error(f"pipeline crashed: {str(ex)}")

if __name__ == "__main__":
    main()
