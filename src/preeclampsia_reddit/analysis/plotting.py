"""Shared figure styling and output helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only ever written to disk

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {"negative": "#EF476F", "neutral": "#FFD166", "positive": "#06D6A0"}
PERIOD_ORDER = ["Pre-COVID", "Post-COVID"]
PERIOD_COLORS = {"Pre-COVID": "#A23B72", "Post-COVID": "#2E86AB"}
PRIMARY = "#2E86AB"
SECONDARY = "#118AB2"
DPI = 300


def apply_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)


def sentiment_colors(categories) -> list[str]:
    return [SENTIMENT_COLORS.get(c, "#999999") for c in categories]


def period_colors(periods) -> list[str]:
    return [PERIOD_COLORS.get(p, "#999999") for p in periods]


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
