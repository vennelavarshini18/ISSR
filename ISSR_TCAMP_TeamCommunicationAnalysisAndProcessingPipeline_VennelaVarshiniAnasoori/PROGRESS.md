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
- moved the project to a new conda environment using python 3.12
- locked exact versions for torch, torchaudio, and speechmos in `requirements.txt` so everyone has the exact same setup
- tested the pipeline locally and deepfilternet gave good scores (stoi: 0.9988, si-sdr: 36.638 db, dnsmos: 3.0126)
- finished phase 1

### June 6: Phase 2 Diarization & Pipeline Setup
- started phase 2 by adding `pyannote.audio` to identify speakers
- the code now tracks who spoke when from the cleaned audio
- added code to calculate diarization error rate (der) to measure accuracy
- cleaned up the repository
- built a single script (`run_pipeline.py`) that runs enhancement, diarization, and evaluation together
- fixed a bug in `speechbrain` that was crashing the tests on windows
- added a feature to automatically find and evaluate against the ground-truth rttm file

### June 13: Phase 2 Evaluation & Edge Cases
- made an `environment.yml` to standardize the local setup
- added `.rttm` file generation and a `--num-speakers` argument
- updated der calculation to include miss, false alarm, and confusion rates
- added `evaluate_batch.py` to test multiple files automatically

### June 16: Fourth Meeting
- discussed about the batch evaluation result on ami sample about deepfilter affecting miss rate metric
- concluded to run a mass batch evaluation on diverse multiple ami samples 

### June 20: Mass Batch Evaluation 
- evaluated 10 standard ami corpus datasets
- ran the evaluation script across all files for raw, noisereduce, and deepfilternet
- added audio chunking in `enhance.py` to fix memory crashes on large audio files
- noticed that `noisereduce` messes up diarization by erasing voice features, which causes the confusion rate to go up
- noticed that `deepfilternet` preserves speaker identity better, but it's a bit too aggressive and suppresses quiet speech, causing the miss rate to go up

### June 21: Tuning for next phase
- moved deepfilternet initialization into an `AudioEnhancer` class in `enhance.py` so the model doesn't reload constantly during batch runs
- lowered pyannote's segmentation threshold to 0.30 to make it more sensitive to the quiet speech that deepfilternet suppresses

### June 23: Fifth Meeting
- discussed midterm expectations.
- decided deepfilternet stays as the primary enhancement model, identified miss rate as the main bottleneck
- agreed to run a 4-step validation experiment across diverse ami samples to find a way to recover quiet speech without increasing false alarms or confusion.
- discussed exploring a post-filter speech normalization method 

### June 25: Phase 2.5 Tuning & Batch Setup
- selected dynamic range compression (drc) using the `pedalboard` package to recover suppressed speech without amplifying background noise
- implemented `_apply_dynamic_range_compression` in the enhancement module using a noisegate and compressor
- modified the batch evaluation script to test 5 specific edge-case ami samples across 5 different pipeline configurations

### June 27: Phase 2.5 Results & Phase 3 Architecture
- completed the 5-step batch validation experiments on the edge-case ami files
- concluded that post-processing cannot recover the quiet speech without breaking diarization
- documented all findings 

### June 28: GPU Migration & Code Cleanup
- recognized that running Pyannote on CPU is a massive bottleneck (taking >30 mins per file)
- migrated the repository execution to google colab (t4 gpu) using a github + google drive hybrid approach

