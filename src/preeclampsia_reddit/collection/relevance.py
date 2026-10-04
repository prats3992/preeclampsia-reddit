"""Keyword matching and relevance scoring for collected posts and comments."""

from __future__ import annotations

from preeclampsia_reddit.config import KEYWORD_WEIGHTS, KEYWORDS_BY_LLM, SUBREDDIT_WEIGHTS

DEFAULT_SUBREDDIT_WEIGHT = 0.3

# For non-dedicated subreddits a post must mention at least one of these to be kept
CORE_TERMS = (
    "preeclampsia", "pre-eclampsia", "pre eclampsia", "eclampsia",
    "hellp", "toxemia", "gestational hypertension",
)


def match_keywords(text: str, llms: list[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Keywords found in ``text`` (case-insensitive substring), grouped by LLM and category."""
    text_lower = text.lower()
    return {
        llm: {
            category: [kw for kw in keywords if kw.lower() in text_lower]
            for category, keywords in KEYWORDS_BY_LLM.get(llm, {}).items()
        }
        for llm in (llms or list(KEYWORDS_BY_LLM))
    }


def count_matches(matched: dict[str, dict[str, list[str]]]) -> int:
    return sum(len(kws) for categories in matched.values() for kws in categories.values())


def has_core_term(text: str) -> bool:
    text_lower = text.lower()
    return any(term in text_lower for term in CORE_TERMS)


def relevance_score(subreddit: str, matched: dict[str, list[str]]) -> float:
    """Score in [0, 1]: 40% subreddit weight + 60% normalised category-weighted keyword score.

    ``matched`` is one LLM's ``{category: [keywords]}`` mapping from :func:`match_keywords`.
    """
    subreddit_weight = SUBREDDIT_WEIGHTS.get(subreddit, {}).get("weight", DEFAULT_SUBREDDIT_WEIGHT)

    keyword_score = 0.0
    keyword_count = 0
    for category, keywords in matched.items():
        if keywords:
            keyword_score += len(keywords) * KEYWORD_WEIGHTS.get(category, 0.5)
            keyword_count += len(keywords)

    keyword_score = min(keyword_score / (keyword_count * 2), 1.0) if keyword_count else 0.0
    return round(subreddit_weight * 0.4 + keyword_score * 0.6, 3)
