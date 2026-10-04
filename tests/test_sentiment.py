import pandas as pd

from preeclampsia_reddit.sentiment import categorize, score_frame, summarize


def test_categorize_thresholds():
    assert categorize(0.05) == "positive"
    assert categorize(-0.05) == "negative"
    assert categorize(0.0) == "neutral"


def test_scores_stay_attached_to_their_rows_with_non_default_index():
    # Regression: scores used to be assigned positionally onto a reordered index,
    # silently giving each post another post's sentiment.
    df = pd.DataFrame(
        {"text": ["I love this, wonderful happy news", "Terrible, awful, horrible day",
                  "The appointment is on Tuesday"]},
        index=[2, 0, 1],
    )
    scored = score_frame(df, "text")
    assert scored.loc[2, "sentiment_category"] == "positive"
    assert scored.loc[0, "sentiment_category"] == "negative"
    assert scored.loc[1, "sentiment_category"] == "neutral"
    assert "sentiment_compound" not in df.columns  # input is not mutated


def test_empty_text_is_neutral_and_summary_reports_counts():
    scored = score_frame(pd.DataFrame({"text": ["", None, "great"]}), "text")
    assert scored["sentiment_category"].tolist() == ["neutral", "neutral", "positive"]
    summary = summarize(scored, pd.DataFrame())
    assert summary["posts"]["total"] == 3 and "comments" not in summary
