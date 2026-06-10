from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings


def create_persistent_client(path: str) -> Any:
    """Create a Chroma persistent client pinned to the default tenant/database.

    Chroma 1.x separates tenant/database. If these are not set explicitly,
    wrappers can end up reading from an empty namespace.
    """
    settings = Settings()
    try:
        return chromadb.PersistentClient(
            path=path,
            settings=settings,
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    except TypeError:
        # Compatibility fallback for older chromadb signatures.
        return chromadb.PersistentClient(path=path, settings=settings)
