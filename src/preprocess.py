import argparse
import os
from pathlib import Path

from src.file_reader import get_full_document_text

SOURCE_MAP = {
    "./RAG_data": "./processed_txt/rag_kb",
}


SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".docx", ".pptx", ".msg", ".txt"}


def _should_process(source_path: Path, out_path: Path, force: bool) -> bool:
    if force or not out_path.exists():
        return True
    return source_path.stat().st_mtime_ns > out_path.stat().st_mtime_ns


def preprocess_files(target: str = "all", force: bool = False):
    print("--- STARTE PREPROCESSING ---")
    target = str(target or "all").strip().lower()
    active_map = SOURCE_MAP if target in {"all", "rag"} else {}
    summary = {
        "processed": 0,
        "skipped": 0,
        "empty": 0,
        "errors": 0,
        "crashes": 0,
    }

    for folder, out_folder in active_map.items():
        source_dir = Path(folder)
        output_dir = Path(out_folder)
        if not source_dir.exists():
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(path for path in source_dir.iterdir() if path.is_file())
        print(f"\nVerarbeite Ordner '{folder}' ({len(files)} Dateien)...")

        for source_path in files:
            if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                summary["skipped"] += 1
                print(f"  [SKIP] {source_path.name}: Format nicht unterstuetzt.")
                continue

            out_path = output_dir / f"{source_path.stem}.txt"
            if not _should_process(source_path, out_path, force=force):
                summary["skipped"] += 1
                print(f"  [SKIP] {source_path.name}: TXT ist aktuell.")
                continue

            try:
                text_content = get_full_document_text(source_path.name, data_folder=folder)
                if not text_content:
                    summary["empty"] += 1
                    print(f"  [EMPTY] {source_path.name}: Inhalt ist leer.")
                elif text_content.startswith("[FEHLER") or text_content.startswith("[LESEFEHLER"):
                    summary["errors"] += 1
                    print(f"  [ERROR] {source_path.name}: {text_content}")
                else:
                    out_path.write_text(text_content, encoding="utf-8")
                    summary["processed"] += 1
                    print(f"  [OK] {source_path.name}")
            except Exception as e:
                summary["crashes"] += 1
                print(f"  [CRASH] {source_path.name}: {e}")

    print(
        "\nPreprocessing fertig: "
        f"{summary['processed']} verarbeitet, "
        f"{summary['skipped']} uebersprungen, "
        f"{summary['empty']} leer, "
        f"{summary['errors']} Fehler, "
        f"{summary['crashes']} Abstuerze."
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the active RAG source corpus into processed_txt outputs.")
    parser.add_argument("--target", choices=["all", "rag"], default="all")
    parser.add_argument("--force", action="store_true", help="Recreate TXT outputs even when they are newer than sources.")
    args = parser.parse_args()
    preprocess_files(target=args.target, force=args.force)
