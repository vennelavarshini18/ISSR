# TCAMP: Team Communication Processing and Analysis

GSoC 2026 Project | HumanAI Foundation | TRIP Lab, University of Alabama

Contributor: Vennela Varshini Anasoori

---

## Overview
In simulator driving studies, teams talk through headsets. These recordings often catch low-frequency room hum, clicks, and echo. **TCAMP** cleans up this audio so speech is clearer for analysis.

Currently completing phase 1: setting up the base repo and audio enhancement module.

---

## Audio Enhancement Module
We support two options to remove background noise:
- `deepfilter`: deep learning approach using deepfilternet3 (full-band 48kHz processing)
- `noisereduce`: simple spectral gating baseline that works well without needing clean reference audio

### Screening Task Results (on 3-min AMI sample)
| Method | SNR (dB) | STOI | DNSMOS | Notes |
|---|---|---|---|---|
| Noisy Input | -2.16 | 0.621 | 1.324 | Baseline room noise |
| Spectral Subtraction | 4.75 | 0.610 | 1.421 | Slight background artifacts |
| Wiener Filter | 1.74 | 0.638 | 1.288 | Minor sound dampening |
| **Noisereduce** | 2.55 | 0.633 | **1.725** | Best overall quality on unlabelled data |
| combined pipeline | -6.87 | 0.582 | 1.275 | audio gets distorted from over-processing |

**Takeaway:** noisereduce gives the highest reference-free quality score (dnsmos), making it a great fallback for real lab recordings.

---

## Setup
Install everything needed for enhancement and evaluation:
```bash
pip install -r requirements.txt
```

---

## Usage
Run enhancement directly in python:
```python
from tcamp.enhance import enhance_audio

results = enhance_audio(
    input_path="input.wav",
    output_path="enhanced.wav",
    method="noisereduce"  # or "deepfilter"
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

## Repo Structure
- `tcamp/enhance/`: audio enhancement scripts and evaluation metrics
- `tests/`: test suite using real lab recordings
- `observations/`: folder where cleaned audio files are saved to listen to later
- `screening_notebooks/`: original screening task work and audio samples
