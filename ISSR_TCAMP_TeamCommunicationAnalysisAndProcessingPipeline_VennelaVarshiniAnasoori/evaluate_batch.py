import os
import glob
import time
from pathlib import Path
from tcamp.pipeline import TCAMPPipeline

def generate_markdown_table(results_list, output_file="evaluation_results.md"):
    with open(output_file, 'w') as f:
        f.write("# TCAMP Batch Evaluation Results\n\n")
        f.write("| Sample | Condition | Expected Speakers | Actual Speakers | DER | Miss | False Alarm | Confusion | Runtime (s) | Notes |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        
        for res in results_list:
            sample_name = Path(res['input_audio']).stem
            condition = res.get("condition", "Unknown")
            runtime = res.get("runtime", 0.0)
            der_stats = res.get("der_breakdown", {})
            der = res.get("der", 0.0)
            if der is None: der = 0.0
            
            # Count actual speakers detected
            import json
            try:
                with open(res['diarization_results'], 'r') as jf:
                    diarization_data = json.load(jf)
                    actual_speakers = len(set(seg['speaker'] for seg in diarization_data))
            except Exception:
                actual_speakers = "?"
            
            miss = der_stats.get("miss", 0.0) if der_stats else 0.0
            fa = der_stats.get("false_alarm", 0.0) if der_stats else 0.0
            conf = der_stats.get("confusion", 0.0) if der_stats else 0.0
            
            if miss is None: miss = 0.0
            if fa is None: fa = 0.0
            if conf is None: conf = 0.0
            
            notes = "Auto-evaluated"
            if miss > 0.15: notes += " (High Miss Rate)"
            if conf > 0.10: notes += " (High Confusion)"
            if fa > 0.10: notes += " (High FA)"
            
            f.write(f"| {sample_name} | **{condition}** | ? | {actual_speakers} | {der:.2%} | {miss:.2%} | {fa:.2%} | {conf:.2%} | {runtime:.1f} | {notes} |\n")

def main():
    print("starting batch evaluation...")
    pipeline = TCAMPPipeline()
    
    audio_dir = Path("screening_notebooks/sample_input_and_output_files")
    
    # Search for all .wav files in the directory
    # Note: Ensure you run download_ami.py first to populate this directory!
    test_files = [f.name for f in audio_dir.glob("*.wav") if "enhanced" not in f.name and "sample_output" not in f.name]
    if not test_files:
        print("no valid ami wav files found. make sure to download them first.")
        return
        
    print(f"found {len(test_files)} files to evaluate: {test_files}")
    
    conditions = [
        {"name": "Raw", "phase": "diarize", "method": "deepfilter"}, # skips enhancement
        {"name": "NoiseReduce", "phase": "all", "method": "noisereduce"},
        {"name": "DeepFilterNet", "phase": "all", "method": "deepfilter"}
    ]
    
    results = []
    
    for filename in test_files:
        audio_path = audio_dir / filename
        print(f"\n==============================================")
        print(f"evaluating {filename} across all conditions...")
        
        for cond in conditions:
            print(f"\n  -> testing condition: {cond['name']}")
            start_time = time.time()
            try:
                res = pipeline.process(
                    input_audio=audio_path,
                    output_dir="observations/outputs",
                    enhance_method=cond["method"],
                    phase=cond["phase"]
                )
                res["condition"] = cond["name"]
                res["runtime"] = time.time() - start_time
                results.append(res)
            except Exception as e:
                print(f"error processing {filename} under {cond['name']}: {e}")
                
    print("\ngenerating markdown report...")
    generate_markdown_table(results)
    print("see evaluation_results.md for the tracking table.")

if __name__ == "__main__":
    main()
