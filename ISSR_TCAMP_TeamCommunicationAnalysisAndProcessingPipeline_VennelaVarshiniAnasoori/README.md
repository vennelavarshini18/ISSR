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

### Screening Task Results (on 3-min AMI sample)
| Method | SNR (dB) | STOI | DNSMOS | Notes |
|---|---|---|---|---|
| Noisy Input | -2.16 | 0.621 | 1.324 | Baseline room noise |
| Spectral Subtraction | 4.75 | 0.610 | 1.421 | Slight background artifacts |
| Wiener Filter | 1.74 | 0.638 | 1.288 | Minor sound dampening |
| **NoiseReduce** | 2.55 | 0.633 | **1.725** | Best overall quality on unlabelled data |

**Takeaway:** NoiseReduce gives the highest reference-free quality score (DNSMOS), making it a great fallback for real lab recordings.

---

## Setup
Install everything needed for enhancement and evaluation. All tools are configured to run completely offline locally (no cloud APIs) as per lab requirements.
```bash
pip install -r requirements.txt
```

---

## Usage
Run enhancement directly in Python, specifying your target sampling rate to ensure metric alignment:
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
