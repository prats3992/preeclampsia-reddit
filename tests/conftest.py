import random
from datetime import datetime, timezone

import pytest

from preeclampsia_reddit.storage import LocalStorage

SUBREDDITS = ["preeclampsia", "BabyBumps", "pregnant", "NICUParents", "beyondthebump"]
POSITIVE = ["so grateful healthy baby home safe", "amazing nurses wonderful support thank you",
            "happy recovery going great love"]
NEGATIVE = ["scared terrible headache awful vision", "worried blood pressure crisis hospital",
            "horrible emergency c-section nicu painful"]
NEUTRAL = ["preeclampsia magnesium sulfate labetalol", "protein in urine 24-hour urine test",
           "induction at 37 weeks blood pressure monitoring"]


def _epoch(year: int, month: int = 6) -> int:
    return int(datetime(year, month, 15, tzinfo=timezone.utc).timestamp())


def make_synthetic_records():
    """~300 posts across 2014-2025 and several subreddits, plus one comment per post."""
    rng = random.Random(0)
    posts, comments = [], []
    for i in range(300):
        year = rng.choice(range(2014, 2026))
        text = rng.choice(rng.choice([POSITIVE, NEGATIVE, NEUTRAL]))
        posts.append({
            "id": f"p{i:04d}",
            "subreddit": rng.choice(SUBREDDITS),
            "title": f"Preeclampsia update {i}",
            "selftext": f"{text} {rng.choice(NEUTRAL)}",
            "created_utc": _epoch(year, rng.randint(1, 12)),
            "score": rng.randint(0, 50),
            "num_comments": 1,
        })
        comments.append({
            "id": f"c{i:04d}",
            "post_id": f"p{i:04d}",
            "subreddit": posts[-1]["subreddit"],
            "body": rng.choice(POSITIVE + NEGATIVE) + " you are not alone",
            "created_utc": posts[-1]["created_utc"] + 3600,
            "score": 1,
        })
    return posts, comments


@pytest.fixture
def synthetic_records():
    return make_synthetic_records()


@pytest.fixture
def local_storage(tmp_path, synthetic_records):
    storage = LocalStorage(tmp_path / "raw")
    posts, comments = synthetic_records
    storage.upsert_posts(posts)
    storage.upsert_comments(comments)
    return storage
