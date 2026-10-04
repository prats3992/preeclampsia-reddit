"""Overall EDA: sentiment distributions, per-subreddit sentiment, TF-IDF, LDA topics and
medical terminology."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

from preeclampsia_reddit.analysis.plotting import (
    PRIMARY,
    SECONDARY,
    SENTIMENT_COLORS,
    apply_style,
    save_figure,
    sentiment_colors,
)
from preeclampsia_reddit.config import KEYWORDS_BY_LLM

log = logging.getLogger(__name__)

MEDICAL_CATEGORIES = ("core_terms", "symptoms", "monitoring", "diagnostic")
RANDOM_STATE = 42


def medical_terms() -> set[str]:
    """Lower-cased clinical keywords from every LLM's medical categories."""
    return {
        kw.lower()
        for categories in KEYWORDS_BY_LLM.values()
        for category, keywords in categories.items()
        if category in MEDICAL_CATEGORIES
        for kw in keywords
    }


def _require(df: pd.DataFrame, *cols: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning("Skipping analysis: missing columns %s", missing)
    return not missing


def _mean_tfidf(corpus: list[str], **vectorizer_kwargs) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", **vectorizer_kwargs)
    matrix = vectorizer.fit_transform(corpus)
    return pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "tfidf_score": np.asarray(matrix.mean(axis=0)).ravel(),
    }).sort_values("tfidf_score", ascending=False)


def _barh(ax, labels, values, color, title, xlabel) -> None:
    ax.barh(range(len(values)), values, color=color, alpha=0.7)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")


class OverallEDA:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.medical_terms = medical_terms()
        apply_style()

    def sentiment_distribution(self, posts: pd.DataFrame, comments: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for col, (label, df, hist_color) in enumerate((("Posts", posts, PRIMARY),
                                                       ("Comments", comments, "#A23B72"))):
            if df.empty or "sentiment_category" not in df.columns:
                continue
            counts = df["sentiment_category"].value_counts()
            axes[0, col].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                             colors=sentiment_colors(counts.index), startangle=90)
            axes[0, col].set_title(f"{label} Sentiment Distribution\n(n={len(df):,})",
                                   fontsize=12, fontweight="bold")

            ax = axes[1, col]
            ax.hist(df["sentiment_compound"], bins=50, color=hist_color, alpha=0.7,
                    edgecolor="black")
            ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
            ax.set_xlabel("Compound Score", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"{label}: Compound Score Distribution", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

        save_figure(fig, self.output_dir / "overall_sentiment_distribution.png")

    def sentiment_by_subreddit(self, posts: pd.DataFrame) -> pd.DataFrame | None:
        if not _require(posts, "subreddit", "sentiment_compound"):
            return None

        by_sub = posts.groupby("subreddit").agg(
            avg_sentiment=("sentiment_compound", "mean"),
            median_sentiment=("sentiment_compound", "median"),
            std_sentiment=("sentiment_compound", "std"),
            avg_pos=("sentiment_pos", "mean"),
            avg_neg=("sentiment_neg", "mean"),
            post_count=("id", "count"),
        ).round(3).reset_index().sort_values("avg_sentiment", ascending=False)
        by_sub.to_csv(self.output_dir / "sentiment_by_subreddit.csv", index=False)

        fig, axes = plt.subplots(2, 1, figsize=(16, 14))

        top_20 = by_sub.nlargest(20, "post_count")
        axes[0].barh(top_20["subreddit"], top_20["avg_sentiment"], alpha=0.7, color=[
            SENTIMENT_COLORS["positive" if x >= 0 else "negative"] for x in top_20["avg_sentiment"]
        ])
        axes[0].axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        axes[0].set_xlabel("Average Sentiment Score", fontsize=12)
        axes[0].set_ylabel("Subreddit", fontsize=12)
        axes[0].set_title("Average Sentiment by Subreddit (Top 20 by Volume)", fontsize=14,
                          fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="x")
        for i, (score, n) in enumerate(zip(top_20["avg_sentiment"], top_20["post_count"])):
            axes[0].text(score, i, f"  n={n}", va="center", fontsize=8)

        top_10 = by_sub.nlargest(10, "post_count")
        x = np.arange(len(top_10))
        width = 0.25
        axes[1].bar(x - width, top_10["avg_pos"], width, label="Positive",
                    color=SENTIMENT_COLORS["positive"], alpha=0.7)
        axes[1].bar(x, top_10["avg_neg"], width, label="Negative",
                    color=SENTIMENT_COLORS["negative"], alpha=0.7)
        axes[1].bar(x + width, top_10["avg_sentiment"], width, label="Compound", color=PRIMARY,
                    alpha=0.7)
        axes[1].set_xlabel("Subreddit", fontsize=12)
        axes[1].set_ylabel("Average Score", fontsize=12)
        axes[1].set_title("Sentiment Components by Subreddit (Top 10 by Volume)", fontsize=14,
                          fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(top_10["subreddit"], rotation=45, ha="right")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        save_figure(fig, self.output_dir / "sentiment_by_subreddit.png")
        return by_sub

    def tfidf(self, posts: pd.DataFrame, text_col: str = "full_text",
              max_features: int = 100) -> pd.DataFrame | None:
        if not _require(posts, text_col):
            return None
        corpus = posts[text_col].dropna().astype(str).tolist()
        if not corpus:
            return None

        scores = _mean_tfidf(corpus, max_features=max_features, min_df=5, max_df=0.8)
        scores.to_csv(self.output_dir / "tfidf_scores.csv", index=False)

        top_30 = scores.head(30)
        fig, ax = plt.subplots(figsize=(12, 10))
        _barh(ax, top_30["term"], top_30["tfidf_score"], PRIMARY, "Top 30 Terms by TF-IDF Score",
              "Average TF-IDF Score")
        ax.set_ylabel("Term", fontsize=12)
        save_figure(fig, self.output_dir / "tfidf_top_terms.png")
        return scores

    def topics(self, posts: pd.DataFrame, text_col: str = "full_text", n_topics: int = 5,
               n_words: int = 10) -> dict | None:
        if not _require(posts, text_col):
            return None
        corpus = posts[text_col].dropna().astype(str).tolist()
        if len(corpus) < 10:
            log.warning("Too few documents (%d) for topic modelling", len(corpus))
            return None

        vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.8, ngram_range=(1, 2),
                                     stop_words="english")
        doc_term = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=RANDOM_STATE,
                                        max_iter=50)
        doc_topics = lda.fit_transform(doc_term)
        topics = {
            f"Topic {i + 1}": [vocab[j] for j in weights.argsort()[-n_words:][::-1]]
            for i, weights in enumerate(lda.components_)
        }
        for name, words in topics.items():
            log.info("%s: %s", name, ", ".join(words))

        labels = [f"{name}: {', '.join(words[:5])}" for name, words in topics.items()]
        fig, ax = plt.subplots(figsize=(14, n_topics * 1.5))
        ax.barh(np.arange(n_topics), doc_topics.mean(axis=0), alpha=0.7,
                color=plt.cm.viridis(np.linspace(0, 1, n_topics)))
        ax.set_yticks(np.arange(n_topics))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Average Topic Weight", fontsize=12)
        ax.set_title("Topic Modeling Results (LDA)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        save_figure(fig, self.output_dir / "topic_modeling.png")

        return {"topics": topics, "model": lda, "vectorizer": vectorizer}

    def subreddit_tfidf(self, posts: pd.DataFrame, subreddits: list[str] | None = None,
                        text_col: str = "full_text", n_terms: int = 20) -> None:
        if not _require(posts, "subreddit", text_col):
            return
        subreddits = subreddits or posts["subreddit"].value_counts().head(5).index.tolist()

        fig, axes = plt.subplots(len(subreddits), 1, figsize=(14, 5 * len(subreddits)),
                                 squeeze=False)
        for i, (ax, subreddit) in enumerate(zip(axes[:, 0], subreddits)):
            corpus = posts.loc[posts["subreddit"] == subreddit, text_col].dropna().astype(str)
            if corpus.empty:
                continue
            try:
                top = _mean_tfidf(corpus.tolist(), max_features=100, min_df=2,
                                  max_df=0.8).head(n_terms)
            except ValueError:  # too few documents for min_df / max_df
                log.warning("Not enough text in r/%s for TF-IDF", subreddit)
                continue
            _barh(ax, top["term"], top["tfidf_score"], plt.cm.Set3(i),
                  f"r/{subreddit} - Top {n_terms} Terms (n={len(corpus):,})", "TF-IDF Score")

        save_figure(fig, self.output_dir / "subreddit_tfidf_comparison.png")

    def medical_term_usage(self, posts: pd.DataFrame,
                           text_col: str = "full_text") -> pd.DataFrame | None:
        if not _require(posts, text_col):
            return None

        counts: Counter = Counter()
        for text in posts[text_col].dropna():
            text_lower = str(text).lower()
            counts.update(term for term in self.medical_terms if term in text_lower)
        if not counts:
            log.warning("No medical terms found")
            return None

        frequency = pd.DataFrame(counts.most_common(50), columns=["term", "count"])
        frequency.to_csv(self.output_dir / "medical_terms_frequency.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        top_30 = frequency.head(30)
        _barh(axes[0], top_30["term"], top_30["count"], SECONDARY,
              "Top 30 Medical Terms by Frequency", "Frequency")

        # Terms are repeated by frequency and re-tokenised, as in the published figure
        medical_text = " ".join(f"{term} " * n for term, n in counts.items())
        cloud = WordCloud(width=800, height=600, background_color="white", colormap="RdPu",
                          max_words=100, relative_scaling=0.5,
                          min_font_size=10).generate(medical_text)
        axes[1].imshow(cloud, interpolation="bilinear")
        axes[1].set_title("Medical Terms Word Cloud", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        save_figure(fig, self.output_dir / "medical_terms_analysis.png")
        return frequency

    def run(self, posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
        log.info("Overall EDA -> %s", self.output_dir)
        self.sentiment_distribution(posts, comments)
        results = {
            "subreddit_sentiment": self.sentiment_by_subreddit(posts),
            "tfidf": self.tfidf(posts),
            "topics": self.topics(posts),
        }
        self.subreddit_tfidf(posts)
        results["medical_terms"] = self.medical_term_usage(posts)
        return results
