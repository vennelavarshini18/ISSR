# observations folder

this directory saves cleaned audio files generated during tests so you can listen to them later and share them in mentor meetings.

## output files saved here:
- `enhanced_noisereduce_ami.wav` -- cleaned with noisereduce baseline
- `enhanced_deepfilter_ami.wav` -- cleaned with deepfilternet3 model

## how to run:
```bash
python -m pytest tests/ -v -s
```
