import os
import glob
from pathlib import Path
from tcamp.pipeline import TCAMPPipeline

def generate_markdown_table(results_list, output_file="batch_results.md"):
    with open(output_file, 'w') as f:
        f.write("# TCAMP Batch Evaluation Results\n\n")
        f.write("| Sample | Expected Speakers | Actual Speakers | DER | Miss | False Alarm | Confusion | Notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        for res in results_list:
            sample_name = Path(res['input_audio']).stem
            der_stats = res.get("der_breakdown", {})
            der = res.get("der", 0.0)
            
            # Count actual speakers detected
            import json
            try:
                with open(res['diarization_results'], 'r') as jf:
                    diarization_data = json.load(jf)
                    actual_speakers = len(set(seg['speaker'] for seg in diarization_data))
            except Exception:
                actual_speakers = "?"
            
            miss = der_stats.get("miss", 0.0)
            fa = der_stats.get("false_alarm", 0.0)
            conf = der_stats.get("confusion", 0.0)
            
            notes = "Auto-evaluated"
            if miss > 0.15: notes += " (High Miss Rate)"
            if conf > 0.10: notes += " (High Confusion - Overlap?)"
            
            f.write(f"| {sample_name} | ? | {actual_speakers} | {der:.2%} | {miss:.2%} | {fa:.2%} | {conf:.2%} | {notes} |\n")

def main():
    print("Initializing Batch Evaluation...")
    pipeline = TCAMPPipeline()
    
    audio_dir = Path("screening_notebooks/sample_input_and_output_files")
    # Only test the actual AMI datasets we downloaded
    test_files = ["EN2002a.wav", "ES2004a.wav", "IS1009a.wav"]
    
    results = []
    
    for filename in test_files:
        audio_path = audio_dir / filename
        if not audio_path.exists():
            print(f"Skipping {filename}: File not found.")
            continue
            
        print(f"\n==============================================")
        print(f"Evaluating {filename}...")
        try:
            # We skip enhancement here just to test diarization directly, or we can run all
            # Since these are long files (30-60 mins), enhancement might take a very long time
            # For quick testing, we will just run diarize phase. 
            # If full pipeline is desired, change phase="all"
            res = pipeline.process(
                input_audio=audio_path,
                output_dir="observations/outputs",
                phase="diarize"
            )
            results.append(res)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print("\nGenerating Markdown Report...")
    generate_markdown_table(results)
    print("Done! See batch_results.md for the tracking table.")

if __name__ == "__main__":
    main()
