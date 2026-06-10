from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rag_fingerprint(rag_kb_dir: str) -> Dict[str, object]:
    files_meta: List[Dict[str, object]] = []
    if not os.path.exists(rag_kb_dir):
        return {"files": [], "fingerprint": "missing-rag-kb-dir"}
    for name in sorted(os.listdir(rag_kb_dir)):
        if not name.endswith(".txt"):
            continue
        full = os.path.join(rag_kb_dir, name)
        stat = os.stat(full)
        files_meta.append(
            {
                "name": name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _file_sha256(full),
            }
        )
    canonical = json.dumps(files_meta, sort_keys=True, ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"files": files_meta, "fingerprint": fingerprint}


def _load_manifest(path: str) -> Dict[str, object] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_manifest(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_rag_index(
    rag_kb_dir: str,
    db_dir: str,
    rebuild_callback,
    auto_reindex: bool = True,
    index_settings: Dict[str, object] | None = None,
) -> Tuple[bool, str]:
    manifest_path = os.path.join(db_dir, "index_manifest.json")
    current = build_rag_fingerprint(rag_kb_dir)
    current_settings = index_settings or {}

    if not auto_reindex:
        return False, "auto_reindex_disabled"

    previous = _load_manifest(manifest_path)
    previous_settings = (previous or {}).get("index_settings", {}) if previous else {}
    should_rebuild = (
        previous is None
        or previous.get("fingerprint") != current.get("fingerprint")
        or previous_settings != current_settings
    )
    if should_rebuild:
        rebuild_callback()
        payload = {
            "fingerprint": current["fingerprint"],
            "files": current["files"],
            "index_settings": current_settings,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest_path, payload)
        return True, "rebuilt"
    return False, "up_to_date"

