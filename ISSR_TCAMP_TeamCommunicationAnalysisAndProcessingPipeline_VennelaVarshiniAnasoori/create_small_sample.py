import soundfile as sf
import os
from pathlib import Path

def create_small_chunk():
    input_file = Path("screening_notebooks/sample_input_and_output_files/EN2002a.wav")
    output_file = Path("observations/small_test_sample.wav")
    
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return

    print("Loading EN2002a.wav...")
    audio, sr = sf.read(input_file)
    
    # Extract 30 seconds (e.g., from 1:00 to 1:30 to ensure there's actual speech)
    start_sample = 60 * sr
    end_sample = 90 * sr
    
    chunk = audio[start_sample:end_sample]
    
    sf.write(output_file, chunk, sr)
    print(f"Saved 30-second test chunk to {output_file}")

if __name__ == "__main__":
    create_small_chunk()
