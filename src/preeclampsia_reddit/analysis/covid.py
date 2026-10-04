"""Pre- vs post-COVID-19 comparison of sentiment, volume, subreddits and vocabulary."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from wordcloud import WordCloud

from preeclampsia_reddit.analysis.plotting import (
    PERIOD_COLORS,
    PERIOD_ORDER,
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
    apply_style,
    period_colors,
    save_figure,
)
from preeclampsia_reddit.config import COVID_START

log = logging.getLogger(__name__)


def _require(df: pd.DataFrame, *cols: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning("Skipping analysis: missing columns %s", missing)
    return not missing


def _ordered_periods(index) -> list[str]:
    return [p for p in PERIOD_ORDER if p in index]


class CovidComparison:
    def __init__(self, output_dir: Path, covid_start: pd.Timestamp = COVID_START):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.covid_start = covid_start
        apply_style()

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (df[df["created_datetime"] < self.covid_start],
                df[df["created_datetime"] >= self.covid_start])

    def sentiment_distribution(self, posts: pd.DataFrame) -> pd.DataFrame | None:
        if not _require(posts, "sentiment_category", "covid_period"):
            return None

        counts = (
            posts.groupby(["covid_period", "sentiment_category"]).size()
            .unstack(fill_value=0)
            .reindex(index=_ordered_periods(posts["covid_period"].unique()),
                     columns=SENTIMENT_ORDER, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind="bar", ax=ax, width=0.7, alpha=0.8,
                    color=[SENTIMENT_COLORS[c] for c in counts.columns])
        ax.set_xlabel("COVID Period", fontsize=12)
        ax.set_ylabel("Number of Posts", fontsize=12)
        ax.set_title("Sentiment Distribution: Pre vs Post COVID", fontsize=14, fontweight="bold")
        ax.legend(title="Sentiment")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=0)
        save_figure(fig, self.output_dir / "sentiment_distribution_comparison.png")

        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        log.info("Sentiment distribution (%%):\n%s", pct.round(2))
        return pct

    def vader_scores(self, posts: pd.DataFrame) -> dict | None:
        if not _require(posts, "sentiment_compound", "covid_period", "created_datetime"):
            return None

        summary = posts.groupby("covid_period").agg({
            "sentiment_compound": ["mean", "median", "std", "count"],
            "sentiment_pos": "mean",
            "sentiment_neg": "mean",
            "sentiment_neu": "mean",
        }).round(3)
        summary.columns = ["_".join(col) for col in summary.columns]

        pre, post = self.split(posts)
        t_stat, p_value = stats.ttest_ind(pre["sentiment_compound"].dropna(),
                                          post["sentiment_compound"].dropna())
        log.info("Pre vs post COVID t-test: t=%.4f, p=%.4f", t_stat, p_value)

        averages = posts.groupby("covid_period")["sentiment_compound"].mean()
        averages = averages.reindex(_ordered_periods(averages.index))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(averages.index, averages.values, color=period_colors(averages.index), alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("COVID Period", fontsize=12)
        ax.set_ylabel("Average Sentiment Score", fontsize=12)
        ax.set_title("Average VADER Score Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for i, value in enumerate(averages.values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
        save_figure(fig, self.output_dir / "vader_score_comparison.png")

        return {"summary": summary, "t_statistic": float(t_stat), "p_value": float(p_value)}

    def post_volume(self, posts: pd.DataFrame) -> pd.Series | None:
        if not _require(posts, "covid_period", "year_month"):
            return None

        totals = posts["covid_period"].value_counts()
        totals = totals.reindex(_ordered_periods(totals.index))
        monthly = posts.groupby(["year_month", "covid_period"]).size().unstack(fill_value=0)
        monthly = monthly[_ordered_periods(monthly.columns)]

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        axes[0].bar(totals.index, totals.values, color=period_colors(totals.index), alpha=0.7,
                    width=0.6)
        axes[0].set_xlabel("COVID Period", fontsize=12)
        axes[0].set_ylabel("Number of Posts", fontsize=12)
        axes[0].set_title("Total Post Volume by COVID Period", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, value in enumerate(totals.values):
            axes[0].text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=11)

        monthly.plot(ax=axes[1], color=period_colors(monthly.columns), linewidth=2, marker="o")
        axes[1].axvline(x=pd.Period(self.covid_start, "M"), color="red", linestyle="--",
                        linewidth=2, label="COVID-19 Start")
        axes[1].set_xlabel("Year-Month", fontsize=12)
        axes[1].set_ylabel("Number of Posts", fontsize=12)
        axes[1].set_title("Monthly Post Volume Over Time", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        save_figure(fig, self.output_dir / "post_volume_comparison.png")
        log.info("Posts by period: %s", totals.to_dict())
        return totals

    def sentiment_score_distributions(self, posts: pd.DataFrame) -> None:
        if not _require(posts, "sentiment_compound", "created_datetime"):
            return
        pre, post = self.split(posts)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].hist([pre["sentiment_compound"], post["sentiment_compound"]], bins=30,
                     label=PERIOD_ORDER, color=period_colors(PERIOD_ORDER), alpha=0.6)
        axes[0].set_xlabel("Sentiment Compound Score", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title("Sentiment Score Distribution (Histogram)", fontsize=14,
                          fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        components = ["sentiment_pos", "sentiment_neg", "sentiment_neu"]
        x = np.arange(len(components))
        width = 0.35
        for offset, (period, frame) in zip((-width / 2, width / 2),
                                           zip(PERIOD_ORDER, (pre, post))):
            axes[1].bar(x + offset, [frame[c].mean() for c in components], width, label=period,
                        color=PERIOD_COLORS[period], alpha=0.7)
        axes[1].set_xlabel("Sentiment Component", fontsize=12)
        axes[1].set_ylabel("Average Score", fontsize=12)
        axes[1].set_title("Sentiment Components Comparison", fontsize=14, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["Positive", "Negative", "Neutral"])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        save_figure(fig, self.output_dir / "sentiment_score_distributions.png")

    def subreddit_distribution(self, posts: pd.DataFrame, top_n: int = 15) -> pd.DataFrame | None:
        if not _require(posts, "subreddit", "covid_period"):
            return None

        top = posts["subreddit"].value_counts().head(top_n).index
        subset = posts[posts["subreddit"].isin(top)]
        shares = pd.crosstab(subset["subreddit"], subset["covid_period"], normalize="columns") * 100
        shares = shares[_ordered_periods(shares.columns)]

        fig, ax = plt.subplots(figsize=(14, 10))
        shares.plot(kind="barh", ax=ax, color=period_colors(shares.columns), alpha=0.7)
        ax.set_xlabel("Percentage (%)", fontsize=12)
        ax.set_ylabel("Subreddit", fontsize=12)
        ax.set_title("Post Distribution Across Top Subreddits (by COVID Period)", fontsize=14,
                     fontweight="bold")
        ax.legend(title="COVID Period", loc="best")
        ax.grid(True, alpha=0.3, axis="x")

        save_figure(fig, self.output_dir / "subreddit_distribution_comparison.png")
        return shares

    def wordclouds(self, posts: pd.DataFrame, text_col: str = "full_text") -> None:
        if not _require(posts, text_col, "created_datetime"):
            return
        pre, post = self.split(posts)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        for ax, label, frame, cmap in ((axes[0], "Pre-COVID", pre, "Blues"),
                                       (axes[1], "Post-COVID", post, "Reds")):
            text = " ".join(frame[text_col].dropna().astype(str))
            if text.strip():
                cloud = WordCloud(width=800, height=400, background_color="white", colormap=cmap,
                                  max_words=100, relative_scaling=0.5,
                                  min_font_size=10).generate(text)
                ax.imshow(cloud, interpolation="bilinear")
                ax.set_title(f"{label} Word Cloud (n={len(frame):,})", fontsize=14,
                             fontweight="bold")
            ax.axis("off")

        save_figure(fig, self.output_dir / "wordcloud_comparison.png")

    def run(self, posts: pd.DataFrame) -> dict:
        log.info("Pre vs post COVID comparison -> %s", self.output_dir)
        results = {"sentiment_distribution": self.sentiment_distribution(posts)}
        results["vader_comparison"] = self.vader_scores(posts)
        results["post_volume"] = self.post_volume(posts)
        self.sentiment_score_distributions(posts)
        results["subreddit_distribution"] = self.subreddit_distribution(posts)
        self.wordclouds(posts)
        return results
