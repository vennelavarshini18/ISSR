# TCAMP: Team Communication Processing and Analysis

GSoC 2026 Project | HumanAI Foundation | TRIP Lab, University of Alabama
Contributor: Vennela Varshini Anasoori
Mentors: Piyush Pawar, Josh White

---

## Overview
In simulator driving studies, teams talk through headsets. These recordings often catch low-frequency room hum, clicks, and echo. **TCAMP** cleans up this audio so speech is clearer for analysis.

Currently completing Phase 1: Setting up the base repository and the audio enhancement module.

---

## Audio Enhancement Module
We support two options to remove background noise. To ensure all evaluation metrics are valid, the pipeline standardizes everything to a `target_sr` (default 16000 Hz) before and after enhancement. The pipeline also includes completely offline, reference-free evaluation (DNSMOS) using a local ONNX runtime.

- `deepfilter`: Deep learning approach using DeepFilterNet3. Automatically resamples its native 48kHz output back to the target sampling rate.
- `noisereduce`: Simple spectral gating baseline that works well without needing clean reference audio.

### Pipeline Local Validation (on 3-min AMI sample)
| Method | SI-SDR (dB) | STOI | DNSMOS | Notes |
|---|---|---|---|---|
| Noisy Input | - | 0.621 | 1.324 | Baseline room noise |
| **NoiseReduce** | 4.846 | 0.863 | 2.257 | Best fast/fallback filter for unlabelled data |
| **DeepFilterNet3** | **36.638** | **0.998** | **3.012** | Superior deep learning enhancement, verified on Python 3.12 |

**Takeaway:** DeepFilterNet provides exceptional clarity (DNSMOS > 3.0), but requires strict Conda/Python 3.12 environment stability. NoiseReduce serves as an excellent fallback.

---

## Setup
To ensure environment stability and proper pre-compiled wheel support for deep learning modules (DeepFilterNet, PyTorch), **Conda** with **Python 3.12** is strictly required.

```bash
# Create and activate environment
conda create -n tcamp python=3.12 -y
conda activate tcamp

# Install exact dependency versions
pip install -r requirements.txt
```

---

## Usage
Run enhancement directly in Python inside your `tcamp` Conda environment. Specify your target sampling rate to ensure metric alignment:
```python
from tcamp.enhance import enhance_audio

results = enhance_audio(
    input_path="input.wav",
    output_path="enhanced.wav",
    method="noisereduce",  # or "deepfilter"
    target_sr=16000        # standardizes frequencies
)
print(results)
```

---

## Testing
To run tests on the real audio sample and save outputs to `observations/`:
```bash
python -m pytest tests/ -v -s
```

---

## Repository Structure
- `tcamp/enhance/`: Audio enhancement scripts and evaluation metrics
- `tests/`: Test suite using real lab recordings
- `observations/`: Folder where cleaned audio files are saved to listen to later
- `screening_notebooks/`: Original screening task work and audio samples
