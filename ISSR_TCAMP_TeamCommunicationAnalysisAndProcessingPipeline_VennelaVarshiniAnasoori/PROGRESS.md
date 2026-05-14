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

### Next steps
- review metric alignment approaches with mentors
- add local dnsmos model for reference-free evaluation
