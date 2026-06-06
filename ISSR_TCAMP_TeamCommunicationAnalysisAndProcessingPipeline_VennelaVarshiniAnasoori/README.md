# TCAMP: Team Communication Processing and Analysis

GSoC 2026 Project | HumanAI Foundation | TRIP Lab, University of Alabama
Contributor: Vennela Varshini Anasoori
Mentors: Piyush Pawar, Josh White

---

## Overview
In simulator driving studies, teams talk through headsets. These recordings often catch low-frequency room hum, clicks, and echo. **TCAMP** cleans up this audio so speech is clearer for analysis, and then diarizes the audio to extract precise "who spoke when" timestamps.

Currently completed Phase 1 (Audio Enhancement) and Phase 2 (Speaker Diarization).

---

## Architecture
The pipeline is divided into two core modules that execute sequentially:

### 1. Audio Enhancement
We support two options to remove background noise. To ensure all evaluation metrics are valid, the pipeline standardizes everything to a `target_sr` (default 16000 Hz). The pipeline also includes offline, reference-free evaluation (DNSMOS) using a local ONNX runtime.
- `deepfilter`: Deep learning approach using DeepFilterNet3. Automatically resamples its native 48kHz output back to the target sampling rate.
- `noisereduce`: Simple spectral gating baseline that works well without needing clean reference audio.

### 2. Speaker Diarization
We leverage `pyannote.audio` to segment the enhanced audio and cluster speaker identities. 
- The module extracts timestamps and maps individual speakers to segments.
- Includes a built-in Diarization Error Rate (DER) evaluator that natively uses `pyannote.metrics` against ground-truth RTTM files.

### Pipeline Local Validation (on 3-min AMI sample)
| Method | SI-SDR (dB) | STOI | DNSMOS | Notes |
|---|---|---|---|---|
| Noisy Input | - | 0.621 | 1.324 | Baseline room noise |
| **NoiseReduce** | 4.846 | 0.863 | 2.257 | Best fast/fallback filter for unlabelled data |
| **DeepFilterNet3** | **36.638** | **0.998** | **3.012** | Superior deep learning enhancement, verified on Python 3.12 |

---

## Setup
To ensure environment stability and proper pre-compiled wheel support for deep learning modules (DeepFilterNet, PyTorch, Pyannote), **Conda** with **Python 3.12** is strictly required. You must also supply a Hugging Face token to access the gated Pyannote models.

```bash
# Create and activate environment
conda create -n tcamp python=3.12 -y
conda activate tcamp

# Install exact dependency versions
pip install -r requirements.txt
```

---

## Usage
We provide a unified, production-ready CLI runner to execute the entire pipeline (Enhancement -> Diarization -> Evaluation) in one command:

```bash
# Run the full pipeline (Enhancement -> Diarization -> Auto-Evaluation)
python run_pipeline.py \
    --input screening_notebooks/sample_input_and_output_files/sample_input.wav \
    --output-dir observations/outputs \
    --enhance-method deepfilter \
    --token YOUR_HF_TOKEN
```

*Note: If a ground truth RTTM file named `{input_filename}_ground_truth.rttm` exists in the `observations` folder, the pipeline will automatically evaluate the Diarization Error Rate (DER).*

---

## Testing
To run the automated test suite across both enhancement and diarization modules:
```bash
python -m pytest tests/ -v -s
```

---

## Repository Structure
- `run_pipeline.py`: Unified CLI entry point for the entire pipeline.
- `tcamp/pipeline.py`: Core logic router that bridges enhancement and diarization.
- `tcamp/enhance/`: Audio enhancement models and audio quality metrics (STOI, DNSMOS).
- `tcamp/diarization/`: Pyannote integration and DER tracking algorithms.
- `tests/`: Pytest suite using real lab recordings.
- `observations/`: Folder where cleaned audio, diarization JSON outputs, and evaluation reports are saved.
- `screening_notebooks/`: Original screening task work and audio samples.
