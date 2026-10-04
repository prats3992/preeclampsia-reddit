"""Storage backends for raw Reddit data."""

from __future__ import annotations

from preeclampsia_reddit.settings import Settings
from preeclampsia_reddit.storage.base import Storage
from preeclampsia_reddit.storage.local import LocalStorage

BACKENDS = ("local", "firebase")


def get_storage(settings: Settings, backend: str | None = None) -> Storage:
    """Return the storage backend named by ``backend`` (default: ``settings.storage_backend``)."""
    backend = (backend or settings.storage_backend).lower()
    if backend == "local":
        return LocalStorage(settings.raw_dir)
    if backend == "firebase":
        from preeclampsia_reddit.storage.firebase import FirebaseStorage

        return FirebaseStorage(settings.firebase_database_url, settings.firebase_credentials)
    raise ValueError(f"Unknown storage backend {backend!r}; expected one of {BACKENDS}.")


__all__ = ["BACKENDS", "LocalStorage", "Storage", "get_storage"]
