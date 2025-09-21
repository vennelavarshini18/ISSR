# ===================================== ENFORCE CACHE DIRECTORY======================================
import os

os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"
# ====================================================================================================
import warnings

import pandas as pd
import soundfile as sf
import torch
from loguru import logger
from pyannote.audio import Pipeline

from src.audio_processing import analyze_tone_intensity, extract_audio
from src.ner import extract_named_entities
from src.sentiment_analysis import analyze_sentiment
from src.transcription import transcribe_audio

warnings.filterwarnings("ignore")  # Set environment variable for PyTorch model cache


# Initialize the pipeline with cache_dir to ensure local caching
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    # use_auth_token="", # we don't need this since we running locally
)
    
# Send pipeline to GPU if available and log device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info("CUDA is available. Using GPU for inference.")
else:
    logger.warning("CUDA is not available. Using CPU for inference.")

pipeline.to(device)

params = {
    "clustering": {
        "method": "average",          # Agglomerative clustering
        "min_cluster_size": 14,        # Avoid tiny false clusters
        "threshold": 0.65,            # Tweak between 0.6-0.7 for ~7 speakers
    },

    "segmentation": {
        "min_duration_off": 0.05,      # Min silence to break segments
    }
}

pipeline.instantiate(params)
logger.info("STARTING DIARIZATION PROCESS.....")

def process_video(
    video_path,
    output_csv_path,
    perform_ner=True,
    perform_sentiment=True,
    perform_tone=True,
    debug=False,
):
    """Process a single video: extract audio, perform speaker identification and diarization, run NER, transcribe, analyze sentiment and tone, save to CSV."""
    audio_path = "temp_audio.wav"
    logger.info(f"Extracting audio from {video_path} to {audio_path}")
    extract_audio(video_path, audio_path)

    # 🔧 FIX 1: num_speakers = 7
    diarization = pipeline(audio_path, num_speakers=7)
    audio, sr = sf.read(audio_path)
    output_dir = "speaker_segments"
    os.makedirs(output_dir, exist_ok=True)
    table_data = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start

        # 🔧 FIX 2: Skip very short segments EARLY
        if duration < 0.5:
            if debug:
                logger.debug(f"Skipping segment {speaker}_{turn.start:.1f}_{turn.end:.1f}.wav: duration {duration:.3f}s < 0.5s")
            continue

        # Only now proceed
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment = audio[start_sample:end_sample]

        output_file = os.path.join(
            output_dir, f"{speaker}_{turn.start:.1f}_{turn.end:.1f}.wav"
        )
        sf.write(output_file, segment, sr)

        # 🔧 FIX 3: Protect transcription with try/except
        try:
            transcription = transcribe_audio(output_file)
            if transcription is None or transcription.strip() == "":
                logger.warning(f"Empty transcription for {output_file} .. skipping")
                transcription = ""
        except Exception as e:
            logger.error(f"Transcription failed for {output_file}: {str(e)}.. skipping")
            transcription = ""

        # Only compute if enabled
        sentiment_score = (
            analyze_sentiment(transcription) if perform_sentiment else None
        )
        tone_intensity = (
            analyze_tone_intensity(output_file, 0, len(segment) / sr)
            if perform_tone
            else None
        )
        named_entities = extract_named_entities(transcription) if perform_ner else None

        row = {
            "speaker_id": speaker,
            "start_time": f"{turn.start:.1f}s",
            "end_time": f"{turn.end:.1f}s",
            "transcribed_content": transcription,
        }
        if perform_ner:
            row["named_entities"] = named_entities
        if perform_sentiment:
            row["sentiment_score"] = sentiment_score
        if perform_tone:
            row["tone_intensity"] = tone_intensity

        table_data.append(row)
        logger.success(
            f"Saved and transcribed: {output_file} (start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker})"
        )

    df = pd.DataFrame(table_data)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved DataFrame to CSV: {output_csv_path}")
    os.remove(audio_path)
    logger.info(f"Removed temporary audio file: {audio_path}")
    return df



def process_all_videos_from_path(
    input_dir,
    output_dir,
    perform_ner=True,
    perform_sentiment=True,
    perform_tone=True,
):
    """Process all video files in the input directory, saving results to output directory."""
    logger.info(f"Processing all videos in directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(input_dir):
        if video_file.lower().endswith(".mp4"):
            video_path = os.path.join(input_dir, video_file)
            output_csv_path = os.path.join(
                output_dir, video_file.replace(".mp4", ".csv")
            )
            try:
                process_video(
                    video_path,
                    output_csv_path,
                    perform_ner,
                    perform_sentiment,
                    perform_tone,
                )
                logger.success(f"Processed: {video_file}")
            except Exception as e:
                logger.error(f"Error processing {video_file}: {str(e)}")
