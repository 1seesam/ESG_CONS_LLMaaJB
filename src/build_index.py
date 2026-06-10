import io
import os
import re
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import logging as transformers_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chroma_client import create_persistent_client

DATA_FOLDER = "./processed_txt/rag_kb"
DB_FOLDER = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 70

MARKER_PATTERNS = [
    re.compile(r"^=== DOKUMENT START:.*$"),
    re.compile(r"^=== DOKUMENT ENDE ===$"),
    re.compile(r"^--- PDF SEITE \d+ ---$"),
]


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_text_for_chunking(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if any(pat.match(line) for pat in MARKER_PATTERNS):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_heading_line(line: str) -> bool:
    value = str(line or "").strip()
    if not value:
        return False
    if value.startswith("#"):
        return True
    if re.match(r"^\u00A7\s*\d+[a-zA-Z]*", value):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", value):
        return True
    if value.endswith(":") and len(value) <= 120:
        return True
    return False


def _fixed_size_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(normalized):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    cleaned = _clean_text_for_chunking(text)
    if not cleaned:
        return []

    lines = cleaned.splitlines()
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        if _is_heading_line(line) and current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current = [line]
        else:
            current.append(line)

    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)

    if not sections:
        sections = [cleaned]

    chunks: List[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(_normalize_ws(section))
            continue
        for piece in _fixed_size_split(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append(_normalize_ws(piece))

    return [chunk for chunk in chunks if chunk]


def build_index(
    data_folder: str = DATA_FOLDER,
    db_folder: str = DB_FOLDER,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    print(f"--- INDEXIERUNG der 'Single Source of Truth' ({data_folder}) ---")
    print(f"Chunk config: size={chunk_size}, overlap={chunk_overlap}")

    files = sorted([f for f in os.listdir(data_folder) if f.endswith(".txt")])
    documents: List[Document] = []
    source_counts: Dict[str, int] = {}

    for file in files:
        path = os.path.join(data_folder, file)
        try:
            text = Path(path).read_text(encoding="utf-8")
            chunks = _split_text(
                text=text,
                chunk_size=max(300, int(chunk_size)),
                chunk_overlap=max(0, int(chunk_overlap)),
            )
            for chunk in chunks:
                source_counts[file] = source_counts.get(file, 0) + 1
                chunk_id = f"{file}::chunk_{source_counts[file]:04d}"
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file, "chunk_id": chunk_id},
                    )
                )
        except Exception as exc:
            print(f"Fehler bei {file}: {exc}")

    print(f"Erstelle Vektoren fuer {len(documents)} Chunks...")
    if not documents:
        raise RuntimeError(f"Keine Chunks erzeugt. Pruefe TXT-Dateien in {data_folder}.")

    transformers_logging.set_verbosity_error()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(db_folder):
        shutil.rmtree(db_folder)

    client = create_persistent_client(db_folder)
    Chroma.from_documents(
        documents,
        embedding,
        collection_name="langchain",
        persist_directory=db_folder,
        client=client,
    )

    print("Index fertig.")


if __name__ == "__main__":
    build_index()
