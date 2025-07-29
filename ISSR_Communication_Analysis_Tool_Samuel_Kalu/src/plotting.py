import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype
from PIL import Image


def plot_speaker_charts(df):
    """
    Generate four charts based on speaker diarization data:
    1. Bar chart of word count per speaker
    2. Timeline chart of speaker activity
    3. Stacked bar chart of sentiment (positive, negative, neutral) per speaker
    4. Bar chart of tone intensity per speaker

    Parameters:
    df (pd.DataFrame): DataFrame with columns 'speaker_id', 'start_time', 'end_time',
                       'transcribed_content', 'named_entities', 'sentiment_score'
    Returns:
    PIL.Image.Image: The generated chart as a PIL image.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Convert time columns to float, handling both string and numeric inputs
    if is_string_dtype(df["start_time"]):
        df["start_time"] = pd.to_numeric(
            df["start_time"].str.replace("s", ""), errors="coerce"
        )
    else:
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")

    if is_string_dtype(df["end_time"]):
        df["end_time"] = pd.to_numeric(
            df["end_time"].str.replace("s", ""), errors="coerce"
        )
    else:
        df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")

    # 1. Word count per speaker
    df["word_count"] = df["transcribed_content"].apply(lambda x: len(str(x).split()))
    word_counts = df.groupby("speaker_id")["word_count"].sum().reset_index()

    # 2. Sentiment distribution (derived from sentiment_score)
    def categorize_sentiment(score):
        if score > 0:
            return {"pos": score, "neg": 0, "neu": 0}
        elif score < 0:
            return {"pos": 0, "neg": abs(score), "neu": 0}
        else:
            return {"pos": 0, "neg": 0, "neu": 1}

    sentiment_data = df["sentiment_score"].apply(categorize_sentiment).apply(pd.Series)
    sentiment_data["speaker_id"] = df["speaker_id"]
    sentiment_scores = (
        sentiment_data.groupby("speaker_id")[["pos", "neg", "neu"]].mean().reset_index()
    )

    # 3. Tone intensity (absolute value of sentiment_score)
    intensity_scores = (
        df.groupby("speaker_id")["sentiment_score"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
    )
    intensity_scores.columns = ["speaker_id", "intensity"]

    # Set up plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))

    # Plot 1: Word count per speaker
    sns.barplot(
        data=word_counts, x="speaker_id", y="word_count", palette="Blues_d", ax=axes[0]
    )
    axes[0].set_title("Number of Words per Speaker")
    axes[0].set_xlabel("Speaker ID")
    axes[0].set_ylabel("Word Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Plot 2: Timeline of speaker activity
    for _, row in df.iterrows():
        axes[1].plot(
            [row["start_time"], row["end_time"]],
            [row["speaker_id"], row["speaker_id"]],
            linewidth=4,
            label=row["speaker_id"]
            if row["start_time"]
            == df[df["speaker_id"] == row["speaker_id"]]["start_time"].min()
            else "",
        )
    axes[1].set_title("Speaker Timeline")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Speaker ID")
    axes[1].legend(title="Speaker ID", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: Stacked bar chart for sentiment
    sentiment_scores_sorted = sentiment_scores.set_index("speaker_id").sort_index()
    axes[2].bar(
        sentiment_scores_sorted.index,
        sentiment_scores_sorted["pos"],
        label="Positive",
        color="mediumseagreen",
    )
    axes[2].bar(
        sentiment_scores_sorted.index,
        sentiment_scores_sorted["neu"],
        bottom=sentiment_scores_sorted["pos"],
        label="Neutral",
        color="gold",
    )
    axes[2].bar(
        sentiment_scores_sorted.index,
        sentiment_scores_sorted["neg"],
        bottom=sentiment_scores_sorted["pos"] + sentiment_scores_sorted["neu"],
        label="Negative",
        color="tomato",
    )
    axes[2].set_title("Sentiment Distribution per Speaker (Stacked)")
    axes[2].set_xlabel("Speaker ID")
    axes[2].set_ylabel("Average Sentiment Score")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(title="Sentiment")

    # Plot 4: Tone intensity per speaker
    sns.barplot(
        data=intensity_scores,
        x="speaker_id",
        y="intensity",
        palette="Reds_d",
        ax=axes[3],
    )
    axes[3].set_title("Tone Intensity per Speaker")
    axes[3].set_xlabel("Speaker ID")
    axes[3].set_ylabel("Average Tone Intensity")
    axes[3].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img
