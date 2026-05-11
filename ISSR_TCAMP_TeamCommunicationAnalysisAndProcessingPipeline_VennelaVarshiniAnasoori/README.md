# ISSR Project 01 — Team Communication Processing and Analysis (TCAMP)
**GSoC 2026 | HumanAI Foundation | TRIP Laboratory, University of Alabama**  
Contributor: Vennela Varshini Anasoori | Mentor: Piyush Pawar, Joshua White, Dr. Andrea Underhill, Dr. Benjamin McManus, Dr. Amanda Hudson, Dr. Despina Stavrinos



---

## Project Overview
TCAMP is a modular pipeline for processing and analyzing team communication 
audio from human-factors simulated environments. Built for TRIP Lab's headset 
recording infrastructure.

---

## Stage 1: Audio Enhancement

**Primary method:** DeepFilterNet3 (DL, full-band 48kHz)  
**Baseline:** NoiseReduce (spectral gating, reference-free)  
**Evaluation:** STOI · DNSMOS · SI-SDR on 3-min AMI IHM sample

### Screening Task Results
| Method | SNR (dB) | STOI | DNSMOS |
|--------|----------|------|--------|
| Noisy baseline | -2.16 | 0.621 | 1.324 |
| Spectral Subtraction | 4.75 | 0.610 | 1.421 |
| Wiener Filter | 1.74 | 0.638 | 1.288 |
| NoiseReduce | 2.55 | 0.633 | 1.725 |
| Combined Pipeline | -6.87 | 0.582 | 1.275 |

**Finding:** NoiseReduce leads on DNSMOS, the only reference-free metric, 
making it most valid for real TRIP Lab recordings. Combined pipeline degrades 
performance due to cascaded over-processing.

### Usage
```python
from tcamp.enhance.enhance import enhance_audio

result = enhance_audio(
    input_path="recording.wav",
    output_path="enhanced.wav",
    method="deepfilter"   # or "noisereduce"
)
print(result)  # {'method': 'deepfilter', 'stoi': 0.84, 'dnsmos': 2.1, 'si_sdr': 12.3}
```

---

## Notebooks
- `notebooks/Dataset_selection_N1.ipynb` — Dataset selection rationale + AMI IHM validation
- `notebooks/Audio_enhancement_N2.ipynb` — Enhancement method comparison + DeepFilterNet3 integration
