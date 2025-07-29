# ===================================== ENFORCE CACHE DIRECTORY======================================
import os

os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"
# ====================================================================================================
import warnings

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """Perform sentiment analysis using VADER and return a composite score from -1 to +1."""
    scores = analyzer.polarity_scores(text)
    sentiment_score = scores["compound"]
    return sentiment_score
