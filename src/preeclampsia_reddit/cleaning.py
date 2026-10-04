"""Load raw records from storage, clean text and derive time / COVID-period columns."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import contractions
import emoji
import pandas as pd

from preeclampsia_reddit.config import COVID_START
from preeclampsia_reddit.storage import Storage

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"http\S+|www\S+|https\S+")
_REDDIT_MARKERS_RE = re.compile(r"\[deleted\]|\[removed\]")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")


def records_to_frame(records: dict[str, dict]) -> pd.DataFrame:
    """Turn an ``{id: record}`` mapping into a DataFrame with an ``id`` column."""
    return pd.DataFrame([{**record, "id": key} for key, record in records.items()])


def clean_text(text) -> str:
    """Expand contractions, demojize, strip URLs / Reddit markers and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = _URL_RE.sub("", text)
    text = _REDDIT_MARKERS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return _REPEATED_PUNCT_RE.sub(r"\1", text)


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse Unix epochs or ISO strings into naive UTC datetimes (unparseable -> NaT)."""
    numeric = pd.to_numeric(values, errors="coerce")
    parsed = pd.to_datetime(numeric, unit="s", utc=True)
    iso = numeric.isna() & values.notna()
    if iso.any():
        parsed[iso] = pd.to_datetime(values[iso], utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "created_utc" not in df.columns:
        return df
    df["created_datetime"] = parse_timestamps(df["created_utc"])
    df["year"] = df["created_datetime"].dt.year
    df["month"] = df["created_datetime"].dt.month
    df["year_month"] = df["created_datetime"].dt.to_period("M")
    df["is_post_covid"] = df["created_datetime"] >= COVID_START
    df["covid_period"] = df["is_post_covid"].map({True: "Post-COVID", False: "Pre-COVID"})
    return df


def _finalise(df: pd.DataFrame, text_col: str, label: str) -> pd.DataFrame:
    before = len(df)
    if text_col in df.columns:
        df = df[df[text_col].str.len() > 0]
    if "created_datetime" in df.columns:
        df = df.sort_values("created_datetime", kind="stable")
    log.info("Cleaned %s: %d -> %d rows", label, before, len(df))
    # A fresh RangeIndex keeps positional and label alignment identical downstream
    return df.reset_index(drop=True)


def clean_posts(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return posts
    df = posts.drop_duplicates(subset=["id"], keep="first").copy()
    df["title_clean"] = df.get("title", pd.Series("", index=df.index)).apply(clean_text)
    df["selftext_clean"] = df.get("selftext", pd.Series("", index=df.index)).apply(clean_text)
    df["full_text"] = (df["title_clean"] + " " + df["selftext_clean"]).str.strip()
    for col in ("score", "num_comments"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return _finalise(_add_time_columns(df), "full_text", "posts")


def clean_comments(comments: pd.DataFrame) -> pd.DataFrame:
    if comments.empty:
        return comments
    df = comments.drop_duplicates(subset=["id"], keep="first").copy()
    df["body_clean"] = df.get("body", pd.Series("", index=df.index)).apply(clean_text)
    if "score" in df.columns:
        df["score"] = df["score"].fillna(0).astype(int)
    return _finalise(_add_time_columns(df), "body_clean", "comments")


def load_and_clean(storage: Storage) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading raw data from %s storage", storage.name)
    posts = records_to_frame(storage.load_posts())
    comments = records_to_frame(storage.load_comments())
    log.info("Loaded %d posts and %d comments", len(posts), len(comments))
    return clean_posts(posts), clean_comments(comments)


def summarize(posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
    summary: dict = {"total_posts": len(posts), "total_comments": len(comments)}
    if posts.empty:
        return summary
    if "subreddit" in posts.columns:
        summary["unique_subreddits"] = int(posts["subreddit"].nunique())
        summary["posts_by_subreddit"] = posts["subreddit"].value_counts().to_dict()
    if "created_datetime" in posts.columns:
        summary["date_range"] = {
            "start": str(posts["created_datetime"].min()),
            "end": str(posts["created_datetime"].max()),
        }
        summary["posts_by_year"] = {
            str(year): int(n) for year, n in posts["year"].value_counts().sort_index().items()
        }
        summary["posts_by_covid_period"] = posts["covid_period"].value_counts().to_dict()
    return summary


def save_cleaned(posts: pd.DataFrame, comments: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not posts.empty:
        posts.to_csv(out_dir / "cleaned_posts.csv", index=False)
    if not comments.empty:
        comments.to_csv(out_dir / "cleaned_comments.csv", index=False)
    with (out_dir / "data_summary.json").open("w") as f:
        json.dump(summarize(posts, comments), f, indent=2, default=str)
    log.info("Saved cleaned data to %s", out_dir)
