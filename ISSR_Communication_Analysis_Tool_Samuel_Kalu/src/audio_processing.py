import warnings

import librosa
import numpy as np
from moviepy.editor import VideoFileClip

warnings.filterwarnings("ignore")


def extract_audio(video_path, audio_path):
    """Extract audio from video file and save as WAV."""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path


def analyze_tone_intensity(audio_path, start_time, end_time):
    """Analyze tone intensity using audio amplitude."""
    y, sr = librosa.load(audio_path, sr=None)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]
    rms = librosa.feature.rms(y=segment)[0]
    intensity = np.mean(rms) if rms.size > 0 else 0
    intensity_normalized = min(intensity / 0.5, 1.0)
    return intensity_normalized
