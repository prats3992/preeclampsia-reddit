from preeclampsia_reddit.collection.relevance import (
    count_matches,
    has_core_term,
    match_keywords,
    relevance_score,
)


def test_match_keywords_is_case_insensitive_and_grouped():
    matched = match_keywords("Diagnosed with PRE-ECLAMPSIA, started Labetalol")
    assert "pre-eclampsia" in matched["claude"]["core_terms"]
    assert "Labetalol" in matched["gemini"]["monitoring"]
    assert set(matched) == {"claude", "gemini", "gpt5"}


def test_count_matches():
    assert count_matches(match_keywords("nothing relevant here")) == 0
    assert count_matches(match_keywords("preeclampsia")) > 0


def test_has_core_term():
    assert has_core_term("HELLP syndrome at 32 weeks")
    assert not has_core_term("just a normal pregnancy update")


def test_relevance_score_combines_subreddit_and_keyword_weights():
    # dedicated subreddit (1.0) with one core term (weight 1.0): 0.4*1.0 + 0.6*(1.0/2)
    assert relevance_score("preeclampsia", {"core_terms": ["preeclampsia"]}) == 0.7
    # no keywords: subreddit component only
    assert relevance_score("BabyBumps", {}) == 0.24
    # unknown subreddits fall back to weight 0.3
    assert relevance_score("unknown_sub", {}) == 0.12
