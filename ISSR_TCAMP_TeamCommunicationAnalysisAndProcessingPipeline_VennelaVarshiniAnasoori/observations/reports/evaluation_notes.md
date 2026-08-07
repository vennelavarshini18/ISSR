# Evaluation Notes & Tracking
---

## Part 1: Phase 1 & 2 Bug Tracking

### Problem: STOI/SI-SDR Metric Issues
- **context:** initially, deepfilternet produced low stoi scores when compared to the 16khz ground-truth audio.
- **cause:** deepfilternet outputs 48khz audio. comparing 48khz output to a 16khz reference breaks the evaluation metric.
- **solution:** added a `target_sr` parameter in `enhance.py` to downsample the output to 16khz before saving. this fixed the stoi metric alignment.

### Problem: SpeechBrain Lazy-Loader Crash
- **context:** pyannote pipeline crashed on windows during testing.
- **cause:** the `speechbrain` module has a lazy-loading issue on windows.
- **solution:** added an import step to pre-load the required modules before the pipeline starts.

### Problem: PyTorch 2.5 and Python 3.14
- **context:** the pipeline crashed on python 3.14 due to missing pytorch wheels.
- **solution:** created an `environment.yml` file to lock the project to python 3.12, ensuring mentors can easily reproduce the setup.

---

## Part 2: Mass Batch Evaluation Observations

### Sample: `EN2001a`
- **der:** 8.26% (raw) | 21.00% (noisereduce) | 10.96% (deepfilternet)
- **miss:** 3.85% (raw) | 9.06% (noisereduce) | 6.96% (deepfilternet)
- **false alarm:** 2.14% (raw) | 2.81% (noisereduce) | 1.62% (deepfilternet)
- **confusion:** 2.27% (raw) | 9.13% (noisereduce) | 2.39% (deepfilternet)
- **Observation:** noisereduce struggles with confusion. deepfilternet avoids confusion and reduces false alarms, but suppresses some speech, increasing the miss rate.
- **Takeaway:** raw has the lowest der, but deepfilternet is the best enhancement option since it keeps confusion low.

### Sample: `EN2002a`
- **der:** 24.73% (raw) | 56.45% (noisereduce) | 30.73% (deepfilternet)
- **miss:** 13.39% (raw) | 28.60% (noisereduce) | 23.76% (deepfilternet)
- **false alarm:** 2.86% (raw) | 1.26% (noisereduce) | 1.18% (deepfilternet)
- **confusion:** 8.48% (raw) | 26.59% (noisereduce) | 5.78% (deepfilternet)
- **Observation:** noisereduce has huge confusion (26.59%). deepfilternet lowers confusion and cuts false alarms in half, but its miss rate jumps heavily.
- **Takeaway:** raw is mathematically better, but deepfilternet is the only viable enhancement because noisereduce breaks speaker identities.

### Sample: `ES2003a`
- **der:** 11.41% (raw) | 60.04% (noisereduce) | 15.20% (deepfilternet)
- **miss:** 4.52% (raw) | 10.61% (noisereduce) | 8.24% (deepfilternet)
- **false alarm:** 4.03% (raw) | 3.51% (noisereduce) | 2.60% (deepfilternet)
- **confusion:** 2.85% (raw) | 45.93% (noisereduce) | 4.36% (deepfilternet)
- **Observation:** noisereduce caused severe speaker clustering issues. deepfilternet successfully lowers false alarms by removing noise, but misses quiet speech.
- **Takeaway:** noisereduce is unusable here. deepfilternet is the best enhancement choice.

### Sample: `ES2004a`
- **der:** 19.36% (raw) | 24.39% (noisereduce) | 25.11% (deepfilternet)
- **miss:** 12.43% (raw) | 19.29% (noisereduce) | 19.46% (deepfilternet)
- **false alarm:** 3.21% (raw) | 2.40% (noisereduce) | 2.02% (deepfilternet)
- **confusion:** 3.73% (raw) | 2.70% (noisereduce) | 3.63% (deepfilternet)
- **Observation:** both enhancement models perform similarly. they drop out quiet speech (high miss) but successfully reduce false alarms.
- **Takeaway:** both enhancements degrade der slightly by suppressing speech.

### Sample: `IB4001`
- **der:** 18.71% (raw) | 52.49% (noisereduce) | 23.07% (deepfilternet)
- **miss:** 6.54% (raw) | 12.54% (noisereduce) | 13.29% (deepfilternet)
- **false alarm:** 5.33% (raw) | 3.32% (noisereduce) | 3.16% (deepfilternet)
- **confusion:** 6.83% (raw) | 36.63% (noisereduce) | 6.62% (deepfilternet)
- **Observation:** noisereduce causes massive confusion. deepfilternet doubles the miss rate but preserves speakers and drops false alarms significantly.
- **Takeaway:** deepfilternet is the safer enhancement alternative.

### Sample: `IN1001`
- **der:** 15.49% (raw) | 53.40% (noisereduce) | 17.08% (deepfilternet)
- **miss:** 6.39% (raw) | 21.07% (noisereduce) | 12.65% (deepfilternet)
- **false alarm:** 3.99% (raw) | 1.63% (noisereduce) | 2.08% (deepfilternet)
- **confusion:** 5.11% (raw) | 30.70% (noisereduce) | 2.35% (deepfilternet)
- **Observation:** deepfilternet cuts both confusion and false alarms in half! however, its miss rate jumps, keeping overall der slightly above raw.
- **Takeaway:** deepfilternet is great if avoiding confusion and false alarms is the priority.

### Sample: `IS1000a`
- **der:** 25.75% (raw) | 47.80% (noisereduce) | 33.10% (deepfilternet)
- **miss:** 10.58% (raw) | 20.65% (noisereduce) | 18.64% (deepfilternet)
- **false alarm:** 9.79% (raw) | 4.72% (noisereduce) | 5.61% (deepfilternet)
- **confusion:** 5.38% (raw) | 22.43% (noisereduce) | 8.85% (deepfilternet)
- **Observation:** raw has a huge false alarm rate (9.79%). both enhancements fix the false alarms by deleting noise, but they over-correct and miss too much real speech.
- **Takeaway:** the enhancements remove too much speech here.

### Sample: `IS1009a`
- **der:** 20.38% (raw) | 38.00% (noisereduce) | 24.88% (deepfilternet)
- **miss:** 8.57% (raw) | 15.01% (noisereduce) | 16.04% (deepfilternet)
- **false alarm:** 5.56% (raw) | 4.87% (noisereduce) | 3.37% (deepfilternet)
- **confusion:** 6.25% (raw) | 18.12% (noisereduce) | 5.47% (deepfilternet)
- **Observation:** deepfilternet keeps confusion low and cuts false alarms, but doubles the miss rate.
- **Takeaway:** deepfilternet is the better enhancement choice.

### Sample: `TS3003a`
- **der:** 18.17% (raw) | 27.43% (noisereduce) | 24.33% (deepfilternet)
- **miss:** 11.48% (raw) | 22.07% (noisereduce) | 19.87% (deepfilternet)
- **false alarm:** 2.80% (raw) | 2.05% (noisereduce) | 1.27% (deepfilternet)
- **confusion:** 3.89% (raw) | 3.30% (noisereduce) | 3.19% (deepfilternet)
- **Observation:** confusion is low across the board. deepfilternet cuts false alarms the most, but suffers from high miss rates.
- **Takeaway:** enhancements suffer from high miss rates compared to raw.

### Sample: `TS3007a`
- **der:** 19.60% (raw) | 43.13% (noisereduce) | 28.03% (deepfilternet)
- **miss:** 6.63% (raw) | 13.16% (noisereduce) | 17.06% (deepfilternet)
- **false alarm:** 6.51% (raw) | 4.75% (noisereduce) | 3.47% (deepfilternet)
- **confusion:** 6.46% (raw) | 25.23% (noisereduce) | 7.49% (deepfilternet)
- **Observation:** noisereduce causes high confusion. deepfilternet cuts false alarms in half but causes a high miss rate.
- **Takeaway:** deepfilternet handles confusion better than noisereduce.

### The "Confusion" Failure of Spectral Gating (NoiseReduce)
- **Observation:** in most ami files, traditional spectral gating (`noisereduce`) degraded the baseline der.
- **Cause:** this is mostly driven by the confusion metric (rising from ~5% up to 45%). noisereduce seems to alter the vocal frequencies while removing noise, which causes pyannote's speaker embeddings to cluster multiple different people into just 2 or 3 speakers.

### The Suppression of Neural Enhancement (DeepFilterNet)
- **Observation:** deepfilternet is much better at preserving speaker identities (keeping confusion low) and reducing false alarms. however, it still slightly degraded the raw baseline der.
- **Cause:** this is mostly driven by the miss rate metric. deepfilternet sometimes treats quiet human speech as noise and suppresses it. this causes pyannote's vad to register false silences.

---

## Part 3: Research Findings & Next Steps

### 1. What We Found
the evaluation showed that cleaning audio for human ears (listening comfort) does not automatically mean better machine diarization. 

raw audio actually had the lowest diarization error rate (der) because pyannote is already trained to handle background noise. both enhancement algorithms degraded the diarization performance slightly, but for different reasons (frequency alteration vs. volume suppression).

### 2. Why Keep Using DeepFilterNet?
if raw audio has a better der, why enhance it at all?
1. **Human Transcription:** researchers have to manually listen to these driving simulator recordings to transcribe exact words and analyze driver cognitive load. raw audio has severe engine hum and road noise, causing massive listening fatigue. deepfilternet is needed to make the audio listenable.
2. **ASR (speech-to-text):** while pyannote (diarization) handles noise well, speech recognition models (like whisper) hallucinate and fail in heavy noise. the audio *must* be cleaned for the transcription phase.

### 3. The Goal of Phase 2.5 (Tuning)
since we must use deepfilternet to clean the audio for humans and asr, we have to deal with its main side effect: the increased miss rate in diarization. 
