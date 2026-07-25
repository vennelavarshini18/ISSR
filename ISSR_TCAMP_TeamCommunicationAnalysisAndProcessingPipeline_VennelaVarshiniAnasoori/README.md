# TCAMP: Team Communication Processing and Analysis

GSoC 2026 Project | HumanAI Foundation | TRIP Lab, University of Alabama
Contributor: Vennela Varshini Anasoori
Mentors: Piyush Pawar, Joshua White

---

## Overview
In simulator driving studies, teams talk through headsets. These recordings often catch low-frequency room hum, clicks, and echo. **TCAMP** cleans up this audio so speech is clearer for analysis, and then diarizes the audio to extract precise "who spoke when" timestamps.

Currently completed Phase 1 (Audio Enhancement), Phase 2 (Speaker Diarization), Phase 2.5 (Tuning & Normalization), Phase 3 (Transcription), and Phase 3.5 (Hallucination Management).

---

## Architecture: The Segment-Level Dual-Path Pipeline
The pipeline is divided into three core modules that execute sequentially, employing a unique "Dual-Path" approach to maximize both diarization accuracy and transcription accuracy.

### 1. Audio Enhancement (DeepFilterNet)
Supports two options to remove background noise (standardized to 16000 Hz). The pipeline includes offline, reference-free evaluation (DNSMOS).
- `deepfilter`: Deep learning approach using DeepFilterNet3.
- `noisereduce`: Simple spectral gating fallback.

### 2. Speaker Diarization (Pyannote on RAW Audio)
Leverages `pyannote.audio` to segment the audio and cluster speaker identities. 
- **The Dual-Path Insight:** Diarization is executed on the **RAW audio** rather than the enhanced audio. Enhancement models like DeepFilterNet sometimes distort or suppress quiet speech, causing Pyannote's VAD to miss speakers. By using RAW audio, precise timestamps are extracted without losing quiet speech segments.
- Includes a built-in Diarization Error Rate (DER) evaluator that natively uses `pyannote.metrics`.

### 3. Smart Transcription (WhisperX)
Transcribes the audio segment-by-segment using WhisperX.
- **Smart Segment Selector:** For every timestamp found by Pyannote, the segment is extracted from *both* the RAW audio and the Enhanced audio. If the RMS energy of the enhanced segment is extremely low compared to the raw audio (indicating the enhancement model over-suppressed it), the RAW audio is passed to WhisperX. Otherwise, the clean Enhanced audio is used.
- **Model Selection (`medium.en`):** The pipeline is explicitly optimized for the `medium.en` model with strict `language="en"` constraints. This prevents multilingual hallucinations and avoids the "hallucinated fluency" (grammar autocorrecting) inherent to larger Whisper architectures.
- This results in highly accurate, verbatim text transcripts optimized for behavioral analytics.

### 4. Quality Control (Multi-Condition QC Tagger)
A custom post-diarization analytics module built to identify Pyannote misattributions.
- **The Problem:** Diarization models struggle during sudden pitch shifts (e.g. mumbling, voice dropping, sudden laughing) and often assign the wrong speaker ID to very short segments.
- **The Solution:** The `qc_tagger` uses `librosa.yin` to calculate the fundamental frequency (F0) baseline for each speaker. It mathematically flags any segment where the pitch suddenly deviates by >30% from the speaker's baseline AND the segment is extremely short (< 1 second). This gives researchers a highly targeted list of suspicious segments for manual review, avoiding the need for LLM post-processing.

---

## Setup
The environment is managed via Conda. You will also need a Hugging Face token (via `$env:HF_TOKEN`) to access Pyannote models.

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate tcamp
```

---

## Usage
Use the CLI to run the full pipeline (Enhancement -> Diarization -> QC -> Transcription):

```bash
# Run the full pipeline with the QC tagger using the default medium.en model
python run_pipeline.py --input screening_notebooks/sample_input_and_output_files/EN2002a.wav --run-qc
```

*Note: You can override the transcription model if needed (e.g., `--transcription-model large-v2 --transcription-device cuda`).*

---

## Testing
To run the automated test suite across both enhancement and diarization modules:
```bash
python -m pytest tests/ -v -s
```

---

## Repository Structure
- `run_pipeline.py`: Unified CLI entry point for the entire pipeline.
- `tcamp/pipeline.py`: Core logic router implementing the Segment-Level Dual-Path architecture.
- `tcamp/enhance/`: Audio enhancement models and audio quality metrics.
- `tcamp/diarization/`: Pyannote integration and DER tracking algorithms.
- `tcamp/transcription/`: WhisperX integration.
- `tcamp/analytics/`: Home to the Multi-Condition QC Tagger and upcoming Phase 4 behavioral metrics.
- `tests/`: Pytest suite using real lab recordings.
- `observations/`: Folder where cleaned audio, diarization JSON outputs, transcripts, QC flags, and evaluation reports are saved.
