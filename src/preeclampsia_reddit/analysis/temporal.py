"""Temporal trends: sentiment and post volume across years."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from preeclampsia_reddit.analysis.plotting import (
    PRIMARY,
    SECONDARY,
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
    apply_style,
    save_figure,
)

log = logging.getLogger(__name__)


def _require(df: pd.DataFrame, *cols: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning("Skipping analysis: missing columns %s", missing)
    return not missing


def _counts_by_year(posts: pd.DataFrame) -> pd.DataFrame:
    """Posts per (year, sentiment category), with every category present as a column."""
    return (
        posts.groupby(["year", "sentiment_category"]).size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )


class TemporalAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_style()

    def sentiment_trend_over_years(self, posts: pd.DataFrame) -> pd.DataFrame | None:
        if not _require(posts, "year", "sentiment_compound"):
            return None

        yearly = posts.groupby("year").agg({
            "sentiment_compound": ["mean", "median", "std", "count"],
            "sentiment_pos": "mean",
            "sentiment_neg": "mean",
            "sentiment_neu": "mean",
        }).round(3)
        yearly.columns = ["_".join(col) for col in yearly.columns]
        yearly = yearly.reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        mean, std = yearly["sentiment_compound_mean"], yearly["sentiment_compound_std"]
        axes[0].plot(yearly["year"], mean, marker="o", linewidth=2, markersize=8, color=PRIMARY)
        axes[0].fill_between(yearly["year"], mean - std, mean + std, alpha=0.3, color=PRIMARY)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Year", fontsize=12)
        axes[0].set_ylabel("Average Sentiment (Compound)", fontsize=12)
        axes[0].set_title("Sentiment Trend Over Years", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        for component, marker, label in (("pos", "o", "Positive"), ("neg", "s", "Negative"),
                                         ("neu", "^", "Neutral")):
            category = {"pos": "positive", "neg": "negative", "neu": "neutral"}[component]
            axes[1].plot(yearly["year"], yearly[f"sentiment_{component}_mean"], marker=marker,
                         label=label, linewidth=2, color=SENTIMENT_COLORS[category])
        axes[1].set_xlabel("Year", fontsize=12)
        axes[1].set_ylabel("Average Score", fontsize=12)
        axes[1].set_title("Sentiment Components Over Years", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(yearly["year"], yearly["sentiment_compound_count"], color=SECONDARY, alpha=0.7)
        axes[2].set_xlabel("Year", fontsize=12)
        axes[2].set_ylabel("Number of Posts", fontsize=12)
        axes[2].set_title("Post Volume Over Years", fontsize=14, fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        save_figure(fig, self.output_dir / "sentiment_trend_over_years.png")
        return yearly

    def post_volume_by_sentiment(self, posts: pd.DataFrame) -> None:
        if not _require(posts, "year", "sentiment_category"):
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        volume = _counts_by_year(posts)
        volume.plot(kind="bar", stacked=True, ax=ax, width=0.8,
                    color=[SENTIMENT_COLORS[c] for c in volume.columns])
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Number of Posts", fontsize=12)
        ax.set_title("Post Volume Over Years (Colored by Sentiment)", fontsize=14,
                     fontweight="bold")
        ax.legend(title="Sentiment", loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

        save_figure(fig, self.output_dir / "post_volume_by_sentiment.png")

    def positive_percentage_by_year(self, posts: pd.DataFrame) -> pd.DataFrame | None:
        if not _require(posts, "year", "sentiment_category"):
            return None

        by_year = _counts_by_year(posts)
        totals = by_year.sum(axis=1)
        result = pd.DataFrame({
            "year": by_year.index,
            "positive_percentage": (by_year["positive"] / totals * 100).round(2).values,
            "total_posts": totals.values,
        })

        fig, ax = plt.subplots(figsize=(14, 8))
        color = SENTIMENT_COLORS["positive"]
        ax.plot(result["year"], result["positive_percentage"], marker="o", linewidth=3,
                markersize=10, color=color)
        ax.fill_between(result["year"], result["positive_percentage"], alpha=0.3, color=color)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Positive Posts (%)", fontsize=12)
        ax.set_title("Percentage of Positive Posts by Year", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        for x, y in zip(result["year"], result["positive_percentage"]):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9)

        save_figure(fig, self.output_dir / "positive_percentage_by_year.png")
        return result

    def yoy_sentiment_change(self, posts: pd.DataFrame) -> pd.DataFrame | None:
        if not _require(posts, "year", "sentiment_compound"):
            return None

        yearly = posts.groupby("year")["sentiment_compound"].mean()
        result = pd.DataFrame({
            "year": yearly.index,
            "avg_sentiment": yearly.values.round(3),
            "yoy_change": yearly.diff().values.round(3),
            "yoy_pct_change": (yearly.pct_change() * 100).values.round(2),
        })

        changes = result.iloc[1:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(changes["year"], changes["yoy_change"], alpha=0.7, color=[
            SENTIMENT_COLORS["positive" if x >= 0 else "negative"] for x in changes["yoy_change"]
        ])
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("YoY Sentiment Change", fontsize=12)
        ax.set_title("Year-over-Year Sentiment Change", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        save_figure(fig, self.output_dir / "yoy_sentiment_change.png")
        return result

    def wordcloud_by_year(self, posts: pd.DataFrame, text_col: str = "full_text",
                          max_years: int | None = 6) -> None:
        if not _require(posts, "year", text_col):
            return

        years = sorted(posts["year"].dropna().unique())
        if max_years:
            step = max(1, len(years) // max_years)  # spread selection across the range
            years = years[::step][:max_years]

        n_cols = 3
        n_rows = (len(years) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows), squeeze=False)
        axes = axes.flatten()

        for ax, year in zip(axes, years):
            year_posts = posts[posts["year"] == year]
            text = " ".join(year_posts[text_col].dropna().astype(str))
            if text.strip():
                cloud = WordCloud(width=800, height=400, background_color="white",
                                  colormap="viridis", max_words=100, relative_scaling=0.5,
                                  min_font_size=10).generate(text)
                ax.imshow(cloud, interpolation="bilinear")
                ax.set_title(f"Year {int(year)} (n={len(year_posts)})", fontsize=12,
                             fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"Year {int(year)}", fontsize=12, fontweight="bold")
        for ax in axes:
            ax.axis("off")

        save_figure(fig, self.output_dir / "wordcloud_by_year.png")

    def run(self, posts: pd.DataFrame) -> dict:
        log.info("Temporal trend analysis -> %s", self.output_dir)
        results = {
            "yearly_sentiment": self.sentiment_trend_over_years(posts),
        }
        self.post_volume_by_sentiment(posts)
        results["positive_percentage"] = self.positive_percentage_by_year(posts)
        results["yoy_change"] = self.yoy_sentiment_change(posts)
        self.wordcloud_by_year(posts)
        return results
