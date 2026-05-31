# observations folder

this directory saves cleaned audio files generated during tests so that we can listen to them later and share them in mentor meetings.

## expected output files generated during evaluation:

- `enhanced_noisereduce_ami.wav` -- ami ihm sample enhanced with noisereduce (stoi: 0.863, si-sdr: 4.846 db, 16000hz)
- `enhanced_deepfilter_ami.wav` -- ami ihm sample enhanced with deepfilternet3 (stoi: 0.9988, si-sdr: 36.638 db, aligned 16000hz)

## inspection instructions
```bash
python -m pytest tests/ -v -s
```
