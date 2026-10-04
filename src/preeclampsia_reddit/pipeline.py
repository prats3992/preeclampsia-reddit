"""End-to-end analysis pipeline: clean -> score sentiment -> analyses -> HTML report."""

from __future__ import annotations

import logging
from pathlib import Path

from preeclampsia_reddit import cleaning, sentiment
from preeclampsia_reddit.analysis.covid import CovidComparison
from preeclampsia_reddit.analysis.overall import OverallEDA
from preeclampsia_reddit.analysis.temporal import TemporalAnalyzer
from preeclampsia_reddit.report import build_report
from preeclampsia_reddit.storage import Storage

log = logging.getLogger(__name__)


class NoDataError(RuntimeError):
    pass


def run_analysis(storage: Storage, processed_dir: Path, results_dir: Path) -> dict:
    processed_dir, results_dir = Path(processed_dir), Path(results_dir)

    log.info("[1/6] Cleaning")
    posts, comments = cleaning.load_and_clean(storage)
    if posts.empty:
        raise NoDataError(
            f"No posts found in {storage.name} storage. Run `collect`, or `sync pull` to "
            "download existing data from Firebase."
        )
    cleaning.save_cleaned(posts, comments, processed_dir)

    log.info("[2/6] Sentiment scoring")
    posts = sentiment.score_posts(posts)
    comments = sentiment.score_comments(comments)
    sentiment.save_scored(posts, comments, processed_dir)

    log.info("[3/6] Temporal trends")
    temporal = TemporalAnalyzer(results_dir / "temporal").run(posts)

    log.info("[4/6] Pre vs post COVID")
    covid = CovidComparison(results_dir / "covid_comparison").run(posts)

    log.info("[5/6] Overall EDA")
    overall = OverallEDA(results_dir / "overall").run(posts, comments)

    log.info("[6/6] HTML report")
    report = build_report(posts, comments, results_dir)

    return {
        "posts": posts,
        "comments": comments,
        "temporal": temporal,
        "covid": covid,
        "overall": overall,
        "report": report,
    }
