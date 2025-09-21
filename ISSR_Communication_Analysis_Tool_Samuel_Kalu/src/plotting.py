import io
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype
from PIL import Image

warnings.filterwarnings("ignore")


def plot_speaker_interaction_network(df, ax, speaker_color_map):
    """
    Generate a network chart of speaker interactions.
    An interaction is defined as one speaker speaking immediately after another.
    """
    df = df.copy().sort_values(by="start_time")
    interactions = []
    if not df.empty:
        last_speaker = df.iloc[0]["speaker_id"]
        for i in range(1, len(df)):
            current_speaker = df.iloc[i]["speaker_id"]
            if current_speaker != last_speaker:
                interactions.append((last_speaker, current_speaker))
            last_speaker = current_speaker

    if not interactions:
        ax.text(
            0.5,
            0.5,
            "No speaker interactions detected.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title("Speaker Interaction Network")
        ax.axis("off")
        return

    G = nx.DiGraph()
    for source, target in interactions:
        if G.has_edge(source, target):
            G[source][target]["weight"] += 1
        else:
            G.add_edge(source, target, weight=1)

    # Use a circular layout with increased scale for more distance
    pos = nx.circular_layout(G, scale=2)

    # Node labels are just the speaker IDs
    labels = {node: node for node in G.nodes()}

    # Draw nodes using the provided color map
    node_colors = [speaker_color_map.get(node, "gray") for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, ax=ax)

    # Draw labels and edges
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=0.5,
        edge_color="grey",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        ax=ax,
    )

    ax.set_title("Speaker Interaction Network")
    ax.axis("off")
    # Adjust axis for proper width and height
    ax.set_xlim(ax.get_xlim()[0] * 1.2, ax.get_xlim()[1] * 1.2)
    ax.set_ylim(ax.get_ylim()[0] * 1.2, ax.get_ylim()[1] * 1.2)


def plot_speaker_charts(df):
    """
    Generate six charts based on speaker diarization data using seaborn for a refined look:
    1. Horizontal bar chart of word count per speaker.
    2. Timeline chart (Gantt style) of speaker activity.
    3. Grouped bar chart of sentiment (positive, negative, neutral) per speaker.
    4. Horizontal bar chart of tone intensity per speaker.
    5. Network chart of speaker interactions.
    6. A legend mapping each speaker to a unique color.

    Parameters:
    df (pd.DataFrame): DataFrame with speaker diarization data.

    Returns:
    PIL.Image.Image: The generated chart as a PIL image.
    """
    df = df.copy()
    if "speaker_id " in df.columns:
        df = df.rename(columns={"speaker_id ": "speaker_id"})
    if "speaker_id" in df.columns and pd.api.types.is_string_dtype(df["speaker_id"]):
        df["speaker_id"] = df["speaker_id"].str.strip()

    # --- Data Pre-processing ---
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

    df["word_count"] = df["transcribed_content"].apply(lambda x: len(str(x).split()))
    word_counts = df.groupby("speaker_id")["word_count"].sum().reset_index()

    def categorize_sentiment(score):
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_category"] = df["sentiment_score"].apply(categorize_sentiment)
    sentiment_distribution = (
        df.groupby(["speaker_id", "sentiment_category"]).size().unstack(fill_value=0)
    )

    intensity_scores = (
        df.groupby("speaker_id")["sentiment_score"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
    )
    intensity_scores.columns = ["speaker_id", "intensity"]

    # --- Create a unified, distinct color map for all speakers ---
    # Ensure consistent numeric ordering like SPEAKER_00, SPEAKER_01, ...
    def speaker_sort_key(s):
        # Try to extract trailing number after an underscore, e.g., SPEAKER_00 -> 0
        try:
            parts = str(s).rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return (0, int(parts[1]))
        except Exception:
            pass
        # Fallback: keep original string ordering
        return (1, str(s))

    unique_speakers = sorted(list(set(df["speaker_id"])), key=speaker_sort_key)
    distinct_colors = [
        "#e6194B",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#42d4f4",
        "#f032e6",
        "#bfef45",
        "#fabed4",
        "#469990",
        "#dcbeff",
        "#9A6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#a9a9a9",
    ]
    speaker_color_map = {
        speaker: distinct_colors[i % len(distinct_colors)]
        for i, speaker in enumerate(unique_speakers)
    }

    # --- Create Legend Labels with Roles ---
    word_counts_for_roles = (
        df.groupby("speaker_id")["word_count"].sum().sort_values(ascending=False)
    )

    narrator = None
    if len(word_counts_for_roles) > 0:
        narrator = word_counts_for_roles.index[0]

    car_driver = None
    if len(word_counts_for_roles) > 1:
        car_driver = word_counts_for_roles.index[1]

    legend_labels = []
    for speaker in unique_speakers:
        if speaker == narrator:
            legend_labels.append(f"{speaker} (narrator)")
        elif speaker == car_driver:
            legend_labels.append(f"{speaker} (car cab driver)")
        else:
            legend_labels.append(speaker)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 22))

    # Define the grid
    ax1 = plt.subplot2grid((3, 2), (0, 0))  # Word Count
    ax2 = plt.subplot2grid((3, 2), (0, 1))  # Timeline
    ax3 = plt.subplot2grid((3, 2), (1, 0))  # Sentiment
    ax4 = plt.subplot2grid((3, 2), (1, 1))  # Intensity
    ax5 = plt.subplot2grid((3, 2), (2, 0))  # Network Chart
    ax6 = plt.subplot2grid((3, 2), (2, 1))  # Speaker Legend

    # 1. Word Count per Speaker (Horizontal)
    sns.barplot(
        data=word_counts,
        y="speaker_id",
        x="word_count",
        ax=ax1,
        orient="h",
        palette=speaker_color_map,
    )
    ax1.set_title("Word Count per Speaker")
    ax1.set_xlabel("Total Words Spoken")
    ax1.set_ylabel("Speaker ID")

    # 2. Speaker Timeline (Gantt Chart)
    # To ensure consistent vertical ordering (SPEAKER_00 at top), use the
    # same `unique_speakers` order and plot bars for each speaker in that order.
    # matplotlib places the first category at the bottom for barh, so we'll
    # reverse the y-axis after plotting so SPEAKER_00 appears at the top.
    # Prepare lists for plotting grouped by speaker so ordering is preserved
    for speaker in unique_speakers:
        segs = df[df["speaker_id"] == speaker]
        for _, row in segs.iterrows():
            ax2.barh(
                speaker,
                width=row["end_time"] - row["start_time"],
                left=row["start_time"],
                color=speaker_color_map.get(row["speaker_id"], "black"),
            )

    ax2.set_title("Speaker Timeline")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Speaker ID")
    # Ensure y-ticks show speakers in the desired order and place SPEAKER_00 at top
    ax2.set_yticks(range(len(unique_speakers)))
    ax2.set_yticklabels(unique_speakers)
    ax2.invert_yaxis()

    # 3. Sentiment Heatmap across time (mean sentiment per speaker per time window)
    # Build a time-axis in seconds and map each speech segment's sentiment to all bins it overlaps.
    try:
        overall_start = df["start_time"].min()
        overall_end = df["end_time"].max()
        if (
            pd.isna(overall_start)
            or pd.isna(overall_end)
            or overall_end <= overall_start
        ):
            raise ValueError("Invalid start/end times for heatmap binning")

        # Determine bin width: prefer 1-second bins but cap total bins for long recordings
        duration = overall_end - overall_start
        max_bins = 200
        if duration <= 0:
            raise ValueError("Non-positive duration for heatmap binning")
        bin_width = 1.0 if duration <= max_bins else float(duration) / max_bins

        # Create bin edges and centers (in seconds)
        bins = np.arange(overall_start, overall_end + bin_width, bin_width)
        n_bins = max(1, len(bins) - 1)
        bin_centers = bins[:-1] + bin_width / 2.0

        # Accumulate sentiment scores for each (bin, speaker)
        from collections import defaultdict

        accum = defaultdict(list)
        for _, row in df.iterrows():
            s = row.get("start_time", None)
            e = row.get("end_time", None)
            sp = row.get("speaker_id", None)
            val = row.get("sentiment_score", None)
            if pd.isna(s) or pd.isna(e) or sp is None or pd.isna(val):
                continue
            # compute bin indices this segment overlaps
            start_idx = int(np.floor((s - overall_start) / bin_width))
            end_idx = int(np.floor((e - overall_start) / bin_width))
            start_idx = max(0, start_idx)
            end_idx = min(n_bins - 1, end_idx)
            for bi in range(start_idx, end_idx + 1):
                accum[(bi, sp)].append(val)

        # Build matrix (n_bins x n_speakers)
        speakers = unique_speakers
        mat = np.full((n_bins, len(speakers)), np.nan)
        for bi in range(n_bins):
            for j, sp in enumerate(speakers):
                vals = accum.get((bi, sp), [])
                if vals:
                    mat[bi, j] = float(np.mean(vals))

        # pivot DataFrame: rows=time bins (seconds), columns=speakers
        pivot = pd.DataFrame(mat, index=bin_centers, columns=speakers)

        # Plot heatmap with speakers on y-axis and time (seconds) on x-axis
        sns.heatmap(
            pivot.T,  # speakers x time
            ax=ax3,
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Mean Sentiment Score"},
            linewidths=0,
            linecolor="lightgray",
            xticklabels=False,
            yticklabels=True,
            vmin=-1.0,
            vmax=1.0,
        )

        # Label formatting: show a few time labels in seconds along the x-axis
        n = pivot.shape[0]
        if n > 0:
            step = max(1, int(n / 8))
            xt_positions = np.arange(0, n, step)
            xt_labels = [f"{int(round(pivot.index[i]))}" for i in xt_positions]
            ax3.set_xticks(xt_positions + 0.5)
            # No rotation for sentiment-by-speaker x-axis labels (keep horizontal)
            ax3.set_xticklabels(xt_labels, rotation=0, ha="center")

        ax3.set_title("Sentiment per speaker over time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Speaker ID")
    except Exception:
        # Fallback: if heatmap fails due to insufficient time data, show distribution bar chart
        sentiment_colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
        sentiment_order = ["Negative", "Neutral", "Positive"]
        existing_sentiments = [
            s for s in sentiment_order if s in sentiment_distribution.columns
        ]
        sentiment_distribution = sentiment_distribution[existing_sentiments]
        plot_colors = [sentiment_colors[c] for c in existing_sentiments]
        sentiment_distribution.plot(
            kind="barh", stacked=False, ax=ax3, color=plot_colors
        )
        ax3.set_title("Sentiment Distribution per Speaker")
        ax3.set_xlabel("Number of Segments")
        ax3.set_ylabel("Speaker ID")
        ax3.legend(title="Sentiment")

    # 4. Tone Intensity (Horizontal)
    sns.barplot(
        data=intensity_scores,
        y="speaker_id",
        x="intensity",
        ax=ax4,
        orient="h",
        palette=speaker_color_map,
    )
    ax4.set_title("Average Tone Intensity per Speaker")
    ax4.set_xlabel("Average Intensity Score")
    ax4.set_ylabel("Speaker ID")

    # 5. Speaker Interaction Network
    plot_speaker_interaction_network(df, ax5, speaker_color_map)

    # 6. Speaker Legend
    ax6.axis("off")
    ax6.set_title("Speaker Legend", pad=20)
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=speaker_color_map[speaker])
        for speaker in unique_speakers
    ]
    ax6.legend(
        legend_patches,
        legend_labels,
        loc="center",
        frameon=True,
        edgecolor="black",
        fontsize=12,
    )

    plt.subplots_adjust(
        top=0.95, bottom=0.05, left=0.1, right=0.95, hspace=0.8, wspace=0.3
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img
