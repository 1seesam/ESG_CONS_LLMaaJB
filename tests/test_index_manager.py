from pathlib import Path

from src.benchmarking.index_manager import ensure_rag_index


def test_ensure_rag_index_rebuild_only_on_change():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_index_manager"
    rag_dir = base / "processed_txt" / "rag_kb"
    db_dir = base / "chroma_db"
    rag_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    doc = rag_dir / "a.txt"
    doc.write_text("v1", encoding="utf-8")

    calls = {"n": 0}

    def rebuild():
        calls["n"] += 1

    rebuilt, reason = ensure_rag_index(str(rag_dir), str(db_dir), rebuild, auto_reindex=True)
    assert rebuilt is True
    assert calls["n"] == 1

    rebuilt, reason = ensure_rag_index(str(rag_dir), str(db_dir), rebuild, auto_reindex=True)
    assert rebuilt is False
    assert calls["n"] == 1

    doc.write_text("v2", encoding="utf-8")
    rebuilt, reason = ensure_rag_index(str(rag_dir), str(db_dir), rebuild, auto_reindex=True)
    assert rebuilt is True
    assert calls["n"] == 2


def test_ensure_rag_index_rebuild_on_index_settings_change():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_index_manager_settings"
    rag_dir = base / "processed_txt" / "rag_kb"
    db_dir = base / "chroma_db"
    rag_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    doc = rag_dir / "a.txt"
    doc.write_text("v1", encoding="utf-8")

    calls = {"n": 0}

    def rebuild():
        calls["n"] += 1

    rebuilt, _ = ensure_rag_index(
        str(rag_dir),
        str(db_dir),
        rebuild,
        auto_reindex=True,
        index_settings={"chunk_size": 800, "chunk_overlap": 120},
    )
    assert rebuilt is True
    assert calls["n"] == 1

    rebuilt, _ = ensure_rag_index(
        str(rag_dir),
        str(db_dir),
        rebuild,
        auto_reindex=True,
        index_settings={"chunk_size": 800, "chunk_overlap": 120},
    )
    assert rebuilt is False
    assert calls["n"] == 1

    rebuilt, _ = ensure_rag_index(
        str(rag_dir),
        str(db_dir),
        rebuild,
        auto_reindex=True,
        index_settings={"chunk_size": 1000, "chunk_overlap": 200},
    )
    assert rebuilt is True
    assert calls["n"] == 2

