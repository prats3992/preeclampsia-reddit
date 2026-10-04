"""Self-contained HTML summary report linking every generated figure."""

from __future__ import annotations

import logging
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

FIGURES: dict[str, list[tuple[str, str]]] = {
    "Temporal Trends": [
        ("temporal/sentiment_trend_over_years.png", "Sentiment Trends Over Years"),
        ("temporal/post_volume_by_sentiment.png", "Post Volume by Sentiment"),
        ("temporal/positive_percentage_by_year.png", "Positive Posts Percentage by Year"),
        ("temporal/yoy_sentiment_change.png", "Year-over-Year Sentiment Change"),
        ("temporal/wordcloud_by_year.png", "Word Clouds by Year"),
    ],
    "COVID-19 Comparison": [
        ("covid_comparison/sentiment_distribution_comparison.png",
         "Sentiment Distribution: Pre vs Post COVID"),
        ("covid_comparison/vader_score_comparison.png", "VADER Score Comparison"),
        ("covid_comparison/post_volume_comparison.png", "Post Volume Comparison"),
        ("covid_comparison/sentiment_score_distributions.png", "Sentiment Score Distributions"),
        ("covid_comparison/subreddit_distribution_comparison.png",
         "Subreddit Distribution Comparison"),
        ("covid_comparison/wordcloud_comparison.png", "Word Cloud Comparison"),
    ],
    "Overall EDA": [
        ("overall/overall_sentiment_distribution.png", "Overall Sentiment Distribution"),
        ("overall/sentiment_by_subreddit.png", "Sentiment by Subreddit"),
        ("overall/tfidf_top_terms.png", "Top TF-IDF Terms"),
        ("overall/topic_modeling.png", "Topic Modeling Results"),
        ("overall/subreddit_tfidf_comparison.png", "Subreddit TF-IDF Comparison"),
        ("overall/medical_terms_analysis.png", "Medical Terms Analysis"),
    ],
}

STYLE = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px;
       background: #f5f5f5; }
.container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px;
             box-shadow: 0 0 10px rgba(0,0,0,0.1); }
h1 { color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }
h2 { color: #118AB2; margin-top: 30px; border-bottom: 2px solid #118AB2; padding-bottom: 5px; }
h3 { color: #06D6A0; margin-top: 20px; }
.metric { display: inline-block; background: #f0f8ff; padding: 15px 25px; margin: 10px;
          border-radius: 8px; border-left: 4px solid #2E86AB; }
.metric-value { font-size: 28px; font-weight: bold; color: #2E86AB; }
.metric-label { font-size: 14px; color: #666; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
th { background: #2E86AB; color: white; }
.timestamp { color: #999; font-size: 12px; text-align: right; }
.image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
              gap: 20px; margin: 20px 0; }
.image-container { border: 1px solid #ddd; padding: 10px; background: #fafafa; }
.image-container img { width: 100%; height: auto; }
.image-caption { text-align: center; margin-top: 10px; font-size: 14px; color: #666; }
"""


def _metric(value, label: str) -> str:
    return (f'<div class="metric"><div class="metric-value">{escape(str(value))}</div>'
            f'<div class="metric-label">{escape(label)}</div></div>')


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _date(posts: pd.DataFrame, fn: str) -> str:
    if "created_datetime" not in posts.columns:
        return "N/A"
    return getattr(posts["created_datetime"], fn)().strftime("%Y-%m-%d")


def _key_findings(posts: pd.DataFrame) -> list[str]:
    findings = []
    if "sentiment_compound" in posts.columns:
        avg = posts["sentiment_compound"].mean()
        tone = "positive" if avg > 0 else "negative"
        findings.append(f"Overall post sentiment is <strong>{tone}</strong> "
                        f"(mean compound score <strong>{avg:.3f}</strong>).")
    if "covid_period" in posts.columns:
        counts = posts["covid_period"].value_counts()
        pre, post = counts.get("Pre-COVID", 0), counts.get("Post-COVID", 0)
        if pre:
            change = (post - pre) / pre * 100
            word = "increased" if change > 0 else "decreased"
            findings.append(f"Post volume {word} by <strong>{abs(change):.1f}%</strong> "
                            "after the onset of COVID-19.")
    if "subreddit" in posts.columns and not posts.empty:
        top = escape(str(posts["subreddit"].value_counts().index[0]))
        findings.append(f"Most active subreddit: <strong>r/{top}</strong>.")
    return findings


def build_report(posts: pd.DataFrame, comments: pd.DataFrame, results_dir: Path) -> Path:
    """Write ``results_dir/report.html``; figure links are relative to that directory."""
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Pre-eclampsia Reddit Analysis</title>",
        f"<style>{STYLE}</style></head><body><div class='container'>",
        "<h1>Pre-eclampsia Reddit Data Analysis Report</h1>",
        f"<p class='timestamp'>Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>",
        "<h2>Executive Summary</h2>",
        _metric(f"{len(posts):,}", "Total Posts"),
        _metric(f"{len(comments):,}", "Total Comments"),
        _metric(posts["subreddit"].nunique() if "subreddit" in posts else "N/A", "Subreddits"),
        _metric(f"{posts['sentiment_compound'].mean():.3f}"
                if "sentiment_compound" in posts else "N/A", "Avg Post Sentiment"),
        "<h2>Data Collection Period</h2>",
        f"<p><strong>Start:</strong> {_date(posts, 'min')} &nbsp; "
        f"<strong>End:</strong> {_date(posts, 'max')}</p>",
    ]

    if "sentiment_category" in posts.columns:
        counts = posts["sentiment_category"].value_counts()
        rows = [[s.capitalize(), f"{counts[s]:,}", f"{counts[s] / len(posts) * 100:.1f}%"]
                for s in ("positive", "neutral", "negative") if s in counts]
        parts += ["<h2>Post Sentiment Distribution</h2>",
                  _table(["Sentiment", "Posts", "Share"], rows)]

    if "subreddit" in posts.columns:
        top = posts["subreddit"].value_counts().head(10)
        rows = [[rank, f"r/{sub}", f"{n:,}", f"{n / len(posts) * 100:.1f}%"]
                for rank, (sub, n) in enumerate(top.items(), 1)]
        parts += ["<h2>Top Subreddits by Post Volume</h2>",
                  _table(["Rank", "Subreddit", "Posts", "Share"], rows)]

    parts.append("<h2>Visualizations</h2>")
    for section, figures in FIGURES.items():
        cards = [
            f"<div class='image-container'><img src='{escape(path)}' alt='{escape(caption)}'>"
            f"<div class='image-caption'>{escape(caption)}</div></div>"
            for path, caption in figures if (results_dir / path).exists()
        ]
        if cards:
            parts += [f"<h3>{escape(section)}</h3><div class='image-grid'>", *cards, "</div>"]

    findings = _key_findings(posts)
    if findings:
        parts += ["<h2>Key Findings</h2><ul>", *(f"<li>{f}</li>" for f in findings), "</ul>"]

    parts += [
        "<h2>Methodology</h2><ul>",
        "<li><strong>Collection:</strong> Reddit API (PRAW), weighted keyword filtering across "
        "pregnancy and health subreddits</li>",
        "<li><strong>Cleaning:</strong> URL / markup removal, contraction expansion, "
        "emoji demojization</li>",
        "<li><strong>Sentiment:</strong> VADER compound score (&ge;0.05 positive, "
        "&le;&minus;0.05 negative)</li>",
        "<li><strong>Text mining:</strong> TF-IDF keyword extraction; LDA topic modelling "
        "(5 topics)</li>",
        "<li><strong>Statistics:</strong> independent t-test on pre- vs post-COVID compound "
        "scores</li>",
        "</ul></div></body></html>",
    ]

    path = results_dir / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    log.info("HTML report written to %s", path)
    return path
