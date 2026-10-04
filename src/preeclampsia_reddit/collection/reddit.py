"""Collect pre-eclampsia posts and comments from Reddit via the official API (PRAW)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from tqdm import tqdm

from preeclampsia_reddit.collection.relevance import (
    DEFAULT_SUBREDDIT_WEIGHT,
    count_matches,
    has_core_term,
    match_keywords,
    relevance_score,
)
from preeclampsia_reddit.config import (
    DATA_COLLECTION_CONFIG,
    KEYWORDS_BY_LLM,
    RATE_LIMIT_SECONDS,
    SUBREDDIT_WEIGHTS,
    get_post_limit_for_subreddit,
)
from preeclampsia_reddit.settings import Settings
from preeclampsia_reddit.storage import Storage

log = logging.getLogger(__name__)

# Comment budget per post grows with how many keywords the post matched
COMMENTS_BASE = 30
COMMENTS_PER_KEYWORD = 5
# Posts (with their comments) written to storage per batch
SAVE_BATCH = 25


def _iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _subreddit_info(name: str) -> tuple[float, list[str], str]:
    info = SUBREDDIT_WEIGHTS.get(name, {})
    return (
        info.get("weight", DEFAULT_SUBREDDIT_WEIGHT),
        info.get("llm", []),
        info.get("focus", "broad"),
    )


class RedditCollector:
    def __init__(self, reddit, storage: Storage):
        self.reddit = reddit
        self.storage = storage

    @classmethod
    def from_settings(cls, settings: Settings, storage: Storage) -> RedditCollector:
        try:
            import praw
        except ImportError as e:
            raise RuntimeError("PRAW is not installed. Run: pip install -e '.[collect]'") from e

        if not (settings.reddit_client_id and settings.reddit_client_secret):
            raise RuntimeError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set (see .env.example)."
            )
        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
        return cls(reddit, storage)

    def collect_posts(self, subreddit_name: str, skip_ids: set[str]) -> list[dict]:
        weight, llms, focus = _subreddit_info(subreddit_name)
        max_posts = get_post_limit_for_subreddit(subreddit_name)
        log.info("r/%s (weight %.1f, focus %s): up to %d posts", subreddit_name, weight, focus,
                 max_posts)

        subreddit = self.reddit.subreddit(subreddit_name)
        if focus == "dedicated":
            # Dedicated subreddits: take everything reachable via top + new listings
            candidates = [*subreddit.top(time_filter="all", limit=max_posts),
                          *subreddit.new(limit=max_posts)]
        else:
            core = dict.fromkeys(kw for llm in llms for kw in KEYWORDS_BY_LLM[llm]["core_terms"])
            query = " OR ".join(list(core)[:3])
            log.info("  search query: %s", query)
            candidates = list(subreddit.search(query, limit=max_posts, time_filter="all"))
        log.info("  retrieved %d candidate posts", len(candidates))

        posts: list[dict] = []
        seen: set[str] = set()
        for post in candidates:
            if post.id in seen or post.id in skip_ids:
                continue
            seen.add(post.id)

            selftext = getattr(post, "selftext", "") or ""
            full_text = f"{post.title} {selftext}"
            if focus != "dedicated" and not has_core_term(full_text):
                continue

            matched = match_keywords(full_text)
            posts.append({
                "id": post.id,
                "subreddit": subreddit_name,
                "title": post.title,
                "selftext": selftext,
                "author": str(post.author) if post.author else "[deleted]",
                "created_utc": int(post.created_utc),
                "created_date": _iso_utc(post.created_utc),
                "score": post.score,
                "num_comments": post.num_comments,
                "url": post.url,
                "permalink": f"https://reddit.com{post.permalink}",
                "subreddit_weight": weight,
                "subreddit_focus": focus,
                "llm_suggested_by": llms,
                "matched_keywords": matched,
                "relevance_score": relevance_score(
                    subreddit_name, matched.get(llms[0] if llms else "claude", {})
                ),
            })
            if len(posts) >= max_posts:
                break

        log.info("  kept %d new posts", len(posts))
        return posts

    def collect_comments(self, post: dict) -> list[dict]:
        limit = min(
            COMMENTS_BASE + COMMENTS_PER_KEYWORD * count_matches(post.get("matched_keywords", {})),
            DATA_COLLECTION_CONFIG["max_comments_per_post"],
        )
        min_length = DATA_COLLECTION_CONFIG["min_comment_length"]

        submission = self.reddit.submission(id=post["id"])
        submission.comments.replace_more(limit=0)

        comments = []
        for comment in submission.comments.list()[:limit]:
            body = getattr(comment, "body", None)
            if not body or len(body) <= min_length:
                continue
            matched = match_keywords(body)
            comments.append({
                "id": comment.id,
                "post_id": post["id"],
                "subreddit": post["subreddit"],
                "body": body,
                "author": str(comment.author) if comment.author else "[deleted]",
                "created_utc": int(comment.created_utc),
                "created_date": _iso_utc(comment.created_utc),
                "score": comment.score,
                "parent_id": comment.parent_id,
                "matched_keywords": matched,
                "relevance_score": relevance_score(post["subreddit"], matched.get("claude", {})),
            })
        return comments

    def run(self, subreddits: list[str] | None = None, collect_comments: bool = True) -> dict:
        """Collect from ``subreddits`` (default: all configured), highest weight first."""
        subreddits = subreddits or list(SUBREDDIT_WEIGHTS)
        subreddits = sorted(subreddits, key=lambda s: _subreddit_info(s)[0], reverse=True)

        existing = self.storage.post_ids()
        log.info("Collecting from %d subreddits into %s storage (%d posts already stored)",
                 len(subreddits), self.storage.name, len(existing))

        totals = {"posts": 0, "comments": 0}
        for name in subreddits:
            try:
                posts = self.collect_posts(name, skip_ids=existing)
            except Exception:
                log.exception("Failed to collect posts from r/%s; skipping", name)
                continue

            # Save in small batches, each post only after its comments, so an interrupted run
            # can be restarted without leaving stored posts whose comments were never fetched
            n_comments = 0
            progress = tqdm(total=len(posts), desc=f"r/{name}", disable=not posts)
            for start in range(0, len(posts), SAVE_BATCH):
                batch = posts[start : start + SAVE_BATCH]
                comments = []
                if collect_comments:
                    for post in batch:
                        try:
                            comments.extend(self.collect_comments(post))
                        except Exception as e:
                            log.warning("Comments for post %s failed: %s", post["id"], e)
                        progress.update()
                else:
                    progress.update(len(batch))
                self.storage.upsert_comments(comments)
                self.storage.upsert_posts(batch)
                existing.update(p["id"] for p in batch)
                n_comments += len(comments)
            progress.close()

            totals["posts"] += len(posts)
            totals["comments"] += n_comments
            if posts:
                log.info("  stored %d posts, %d comments", len(posts), n_comments)
            time.sleep(RATE_LIMIT_SECONDS)

        self.storage.record_run({"subreddits": subreddits, "collected": totals})
        log.info("Collection finished: %d new posts, %d comments", totals["posts"],
                 totals["comments"])
        return totals
