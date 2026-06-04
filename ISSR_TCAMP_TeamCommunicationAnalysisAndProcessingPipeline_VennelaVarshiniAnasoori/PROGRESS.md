# Progress Log

## Phase 1: Basic Module Setup (weeks 1-2)
Goal: set up the repository structure and test out early audio enhancement models.

### May 9: Initial Meeting
- met with mentors to discuss goals and workflow
- agreed to keep code simple and easy to test with different models later
- updates will be shared regularly through mails and weekly/biweekly meetings

### May 11: Module Layout
- moved screening notebooks into `screening_notebooks/` to keep them safe
- built basic enhancement tools in `tcamp/enhance/`:
  - `enhance.py`: routes audio to the selected model
  - `baseline.py`: runs simple options like noisereduce
  - `metrics.py`: calculates audio scores (stoi, si-sdr)

### May 14: Testing and Updates
- added complete requirements list including deepfilternet and compatible torch versions
- added pytest suite to test code using real headset samples
- noticed that deepfilternet automatically converts audio to 48kHz, which causes unaligned evaluation scores when directly compared to 16kHz references. will discuss alignment options with mentors next week.

### May 26: Second Meeting
- had our second meeting with mentors Josh and Piyush.
- discussed the sampling rate mismatch issue (16khz input vs 48khz deepfilter output).
- mentors advised standardizing the sampling rate across the entire pipeline.
- confirmed with mentors that everything must run locally (no cloud apis).

### May 31: Pipeline Standardization
- implemented `target_sr` in the enhancement router. deepfilternet outputs are now automatically resampled back to 16000 hz before saving and evaluation.
- metric alignment is fully fixed, deepfilter stoi score jumped to 0.9988 on the aligned scale.

### June 1: Local DNSMOS Integration
- successfully integrated the `speechmos` library for completely offline, local DNSMOS evaluation.
- updated the test suite to calculate and record DNSMOS scores alongside STOI and SI-SDR.
- the pipeline is now fully compliant with our strict local execution req (zero cloud APIs).

### June 2: Third Meeting
- showed the pipeline to mentors, but deepfilter didn't run.
- found out it was because python 3.14 doesn't support the required PyTorch wheels yet.
- mentors told to fix the environment first and switch to a Conda environment with Python 3.12.

### June 4: Fixing the Environment (Phase 1 Complete)
- moved the project to a new Conda environment using Python 3.12.
- locked exact versions for torch, torchaudio, and speechmos in `requirements.txt` so everyone has the exact same setup.
- tested the pipeline locally and everything works, DeepFilterNet ran successfully and gave great scores (STOI: 0.9988, SI-SDR: 36.638 dB, DNSMOS: 3.0126).
- Phase 1 is now fully complete and documented.

### Next steps
- commit Phase 1 code and show the working pipeline to mentors.
- start Phase 2 by adding `pyannote.audio` for speaker diarization.
