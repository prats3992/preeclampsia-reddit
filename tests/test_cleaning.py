import pandas as pd

from preeclampsia_reddit.cleaning import clean_posts, clean_text, parse_timestamps


def test_clean_text():
    raw = "I can't believe it!!! 😀 see https://example.com [deleted]   ok"
    cleaned = clean_text(raw)
    assert "cannot" in cleaned or "can not" in cleaned
    assert "http" not in cleaned and "[deleted]" not in cleaned
    assert "!!" not in cleaned and "  " not in cleaned
    assert "grinning" in cleaned
    assert clean_text(None) == ""


def test_parse_timestamps_handles_epochs_and_iso_in_utc():
    parsed = parse_timestamps(pd.Series([1583020800, "2020-02-29T23:59:59+00:00", None]))
    assert parsed[0] == pd.Timestamp("2020-03-01 00:00:00")
    assert parsed[1] == pd.Timestamp("2020-02-29 23:59:59")
    assert pd.isna(parsed[2])


def test_clean_posts_sorts_by_date_resets_index_and_flags_covid():
    posts = pd.DataFrame([
        {"id": "b", "title": "later", "selftext": "", "created_utc": 1600000000, "score": None},
        {"id": "a", "title": "earlier", "selftext": "", "created_utc": 1500000000, "score": 3},
        {"id": "a", "title": "earlier", "selftext": "", "created_utc": 1500000000, "score": 3},
        {"id": "c", "title": "", "selftext": "", "created_utc": 1550000000, "score": 1},
    ])
    cleaned = clean_posts(posts)
    assert cleaned["id"].tolist() == ["a", "b"]  # deduplicated, empty dropped, date-sorted
    assert cleaned.index.tolist() == [0, 1]
    assert cleaned["covid_period"].tolist() == ["Pre-COVID", "Post-COVID"]
    assert cleaned["score"].tolist() == [3, 0]
    assert cleaned["year"].tolist() == [2017, 2020]
