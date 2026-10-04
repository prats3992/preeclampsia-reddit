"""Local JSON-file storage: one ``<kind>.json`` file per collection under a directory."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from preeclampsia_reddit.storage.base import KINDS, Storage


class LocalStorage(Storage):
    name = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, kind: str) -> Path:
        return self.root / f"{kind}.json"

    def _load(self, kind: str) -> dict[str, dict]:
        path = self.path(kind)
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _upsert(self, kind: str, records: dict[str, dict]) -> None:
        data = self._load(kind)
        data.update(records)
        self._write(kind, data)

    def _delete_all(self) -> None:
        for kind in KINDS:
            self.path(kind).unlink(missing_ok=True)

    def _write(self, kind: str, data: dict) -> None:
        # Write to a temp file then rename, so an interrupted run never leaves a corrupt file
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path(kind))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
