"""Backend-agnostic storage interface for collected Reddit posts and comments.

Records are stored as ``{id: record}`` mappings, mirroring the Firebase Realtime Database
layout, so data can be copied between backends without transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone

POSTS = "posts"
COMMENTS = "comments"
RUNS = "runs"
KINDS = (POSTS, COMMENTS, RUNS)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage(ABC):
    """Abstract store. Subclasses implement the three primitive operations below."""

    name: str

    @abstractmethod
    def _load(self, kind: str) -> dict[str, dict]: ...

    @abstractmethod
    def _upsert(self, kind: str, records: dict[str, dict]) -> None: ...

    @abstractmethod
    def _delete_all(self) -> None: ...

    def load_posts(self) -> dict[str, dict]:
        return self._load(POSTS)

    def load_comments(self) -> dict[str, dict]:
        return self._load(COMMENTS)

    def load_runs(self) -> dict[str, dict]:
        return self._load(RUNS)

    def upsert_posts(self, posts: Iterable[dict], stamp: bool = True) -> int:
        return self._upsert_records(POSTS, posts, stamp)

    def upsert_comments(self, comments: Iterable[dict], stamp: bool = True) -> int:
        return self._upsert_records(COMMENTS, comments, stamp)

    def record_run(self, metadata: dict) -> None:
        key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._upsert(RUNS, {key: {**metadata, "recorded_at": _utcnow()}})

    def clear(self) -> None:
        self._delete_all()

    def post_ids(self) -> set[str]:
        return set(self.load_posts())

    def stats(self) -> dict:
        posts = self.load_posts()
        return {
            "backend": self.name,
            "total_posts": len(posts),
            "total_comments": len(self.load_comments()),
            "posts_by_subreddit": dict(
                Counter(p.get("subreddit", "unknown") for p in posts.values()).most_common()
            ),
        }

    def _upsert_records(self, kind: str, records: Iterable[dict], stamp: bool) -> int:
        now = _utcnow()
        batch = {}
        for record in records:
            record = dict(record)
            if stamp:
                record["stored_at"] = now
            batch[str(record["id"])] = record
        if batch:
            self._upsert(kind, batch)
        return len(batch)
