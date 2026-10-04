"""Runtime settings resolved from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_USER_AGENT = "preeclampsia-reddit-research/1.0"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip("'\"")
    return value or default


@dataclass(frozen=True)
class Settings:
    storage_backend: str = "local"
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    firebase_credentials: Path | None = None
    firebase_database_url: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = DEFAULT_USER_AGENT

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load_settings() -> Settings:
    load_dotenv()
    credentials = _env("FIREBASE_CREDENTIALS") or _env("GOOGLE_APPLICATION_CREDENTIALS")
    return Settings(
        storage_backend=(_env("STORAGE_BACKEND", "local") or "local").lower(),
        data_dir=Path(_env("DATA_DIR", "data")),
        results_dir=Path(_env("RESULTS_DIR", "results")),
        firebase_credentials=Path(credentials) if credentials else None,
        firebase_database_url=_env("FIREBASE_DATABASE_URL"),
        reddit_client_id=_env("REDDIT_CLIENT_ID"),
        reddit_client_secret=_env("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=_env("REDDIT_USER_AGENT", DEFAULT_USER_AGENT),
    )
