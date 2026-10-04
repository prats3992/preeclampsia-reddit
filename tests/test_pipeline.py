import pytest

from preeclampsia_reddit.pipeline import NoDataError, run_analysis
from preeclampsia_reddit.report import FIGURES
from preeclampsia_reddit.storage import LocalStorage


def test_full_pipeline_generates_all_outputs(tmp_path, local_storage):
    results_dir, processed_dir = tmp_path / "results", tmp_path / "processed"
    result = run_analysis(local_storage, processed_dir, results_dir)

    assert len(result["posts"]) == 300
    for figures in FIGURES.values():
        for path, _ in figures:
            assert (results_dir / path).exists(), path
    for name in ("sentiment_by_subreddit.csv", "tfidf_scores.csv", "medical_terms_frequency.csv"):
        assert (results_dir / "overall" / name).exists()
    for name in ("cleaned_posts.csv", "posts_with_sentiment.csv", "sentiment_summary.json"):
        assert (processed_dir / name).exists()

    html = (results_dir / "report.html").read_text()
    assert "Total Posts" in html and "temporal/sentiment_trend_over_years.png" in html
    assert len(result["overall"]["topics"]["topics"]) == 5


def test_pipeline_without_data_raises(tmp_path):
    with pytest.raises(NoDataError):
        run_analysis(LocalStorage(tmp_path / "empty"), tmp_path / "p", tmp_path / "r")
