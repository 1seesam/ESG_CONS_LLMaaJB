import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _reexec_with_project_venv_if_needed() -> None:
    venv_python = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else PROJECT_ROOT / ".venv" / "bin" / "python"
    )
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python == target_python:
        return

    if os.environ.get("THESIS_INGEST_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["THESIS_INGEST_REEXEC"] = "1"
    completed = subprocess.run([str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env, cwd=str(PROJECT_ROOT))
    raise SystemExit(completed.returncode)


def _ensure_project_cwd() -> None:
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass


if __name__ == "__main__":
    _ensure_project_cwd()
    _reexec_with_project_venv_if_needed()

from src.benchmarking.index_manager import build_rag_fingerprint
from src.build_index import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, EMBEDDING_MODEL, build_index
from src.preprocess import preprocess_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert new RAG source documents to TXT and rebuild the Chroma index."
    )
    parser.add_argument("--target", choices=["rag", "all"], default="rag")
    parser.add_argument("--force-preprocess", action="store_true", help="Recreate all TXT files before indexing.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Only rebuild the index from existing TXT files.")
    parser.add_argument("--skip-index", action="store_true", help="Only create/update TXT files; do not rebuild Chroma.")
    parser.add_argument("--rag-kb-dir", default="./processed_txt/rag_kb")
    parser.add_argument("--db-dir", default="./chroma_db")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use locally cached HuggingFace files and clear proxy variables for this process.",
    )
    return parser.parse_args()


def _enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
        os.environ[key] = ""


def _write_index_manifest(rag_kb_dir: str, db_dir: str, chunk_size: int, chunk_overlap: int) -> None:
    payload = build_rag_fingerprint(rag_kb_dir)
    payload.update(
        {
            "index_settings": {
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "embedding_model": EMBEDDING_MODEL,
            },
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    manifest_path = Path(db_dir) / "index_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest geschrieben: {manifest_path}")


def main() -> None:
    args = parse_args()

    if args.offline:
        _enable_offline_mode()
        print("Offline-Modus aktiv: HuggingFace nutzt lokale Cache-Dateien.")

    if not args.skip_preprocess:
        preprocess_files(target=args.target, force=args.force_preprocess)

    if args.skip_index:
        print("Indexierung uebersprungen (--skip-index).")
        return

    build_index(
        data_folder=args.rag_kb_dir,
        db_folder=args.db_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    _write_index_manifest(
        rag_kb_dir=args.rag_kb_dir,
        db_dir=args.db_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("RAG-Ingest fertig.")


if __name__ == "__main__":
    main()
