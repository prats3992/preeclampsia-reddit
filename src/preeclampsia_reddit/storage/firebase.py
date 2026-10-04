"""Firebase Realtime Database storage.

Node names match the database used for the original study, so existing data is read as-is.
"""

from __future__ import annotations

from pathlib import Path

from preeclampsia_reddit.storage.base import COMMENTS, POSTS, RUNS, Storage

NODES = {POSTS: "reddit_posts", COMMENTS: "reddit_comments", RUNS: "collection_metadata"}
# Written by earlier versions of the collector; removed by clear() as well
LEGACY_NODES = ("llm_comparison_analysis",)
# Keep multi-path updates well below Firebase's request size limit
UPSERT_CHUNK = 500


class FirebaseStorage(Storage):
    name = "firebase"

    def __init__(self, database_url: str | None, credentials_path: Path | None = None):
        try:
            import firebase_admin
            from firebase_admin import credentials, db
        except ImportError as e:
            raise RuntimeError(
                "Firebase support is not installed. Run: pip install -e '.[firebase]'"
            ) from e

        if not database_url:
            raise RuntimeError("FIREBASE_DATABASE_URL is not set (see .env.example).")

        if not firebase_admin._apps:
            if credentials_path:
                if not Path(credentials_path).exists():
                    raise RuntimeError(f"Firebase credentials file not found: {credentials_path}")
                cred = credentials.Certificate(str(credentials_path))
            else:
                cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})

        self._root = db.reference()

    def _load(self, kind: str) -> dict[str, dict]:
        data = self._root.child(NODES[kind]).get()
        if isinstance(data, list):  # Firebase returns arrays for integer-like keys
            data = {str(i): v for i, v in enumerate(data) if v is not None}
        return data or {}

    def _upsert(self, kind: str, records: dict[str, dict]) -> None:
        node = NODES[kind]
        items = list(records.items())
        for start in range(0, len(items), UPSERT_CHUNK):
            chunk = items[start : start + UPSERT_CHUNK]
            self._root.update({f"{node}/{key}": value for key, value in chunk})

    def _delete_all(self) -> None:
        for node in (*NODES.values(), *LEGACY_NODES):
            self._root.child(node).delete()
