"""VADER sentiment scoring for posts and comments."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

# Standard VADER compound-score thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
NEUTRAL_SCORES = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def polarity_scores(text) -> dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return dict(NEUTRAL_SCORES)
    return _get_analyzer().polarity_scores(text)


def categorize(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def score_frame(df: pd.DataFrame, text_col: str, desc: str = "Scoring") -> pd.DataFrame:
    """Return a copy of ``df`` with ``sentiment_{neg,neu,pos,compound,category}`` columns."""
    if df.empty or text_col not in df.columns:
        return df
    scores = pd.DataFrame(
        [polarity_scores(t) for t in tqdm(df[text_col], desc=desc)],
        index=df.index,  # keep scores attached to the row they were computed from
    )
    df = df.copy()
    for col in ("neg", "neu", "pos", "compound"):
        df[f"sentiment_{col}"] = scores[col]
    df["sentiment_category"] = df["sentiment_compound"].apply(categorize)
    return df


def score_posts(posts: pd.DataFrame) -> pd.DataFrame:
    text_col = "full_text" if "full_text" in posts.columns else "title_clean"
    return score_frame(posts, text_col, desc="Scoring posts")


def score_comments(comments: pd.DataFrame) -> pd.DataFrame:
    return score_frame(comments, "body_clean", desc="Scoring comments")


def _describe(df: pd.DataFrame) -> dict | None:
    if df.empty or "sentiment_compound" not in df.columns:
        return None
    categories = df["sentiment_category"].value_counts()
    return {
        "total": len(df),
        "avg_compound": float(df["sentiment_compound"].mean()),
        "median_compound": float(df["sentiment_compound"].median()),
        "std_compound": float(df["sentiment_compound"].std()),
        "sentiment_distribution": categories.to_dict(),
        "sentiment_percentages": (categories / len(df) * 100).round(2).to_dict(),
    }


def summarize(posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
    return {
        name: stats
        for name, stats in (("posts", _describe(posts)), ("comments", _describe(comments)))
        if stats is not None
    }


def save_scored(posts: pd.DataFrame, comments: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not posts.empty:
        posts.to_csv(out_dir / "posts_with_sentiment.csv", index=False)
    if not comments.empty:
        comments.to_csv(out_dir / "comments_with_sentiment.csv", index=False)
    with (out_dir / "sentiment_summary.json").open("w") as f:
        json.dump(summarize(posts, comments), f, indent=2, default=str)
    log.info("Saved sentiment-scored data to %s", out_dir)
