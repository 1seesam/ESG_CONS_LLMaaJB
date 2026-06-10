from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarking.index_manager import ensure_rag_index
from src.benchmarking.retrieval import RetrievalAdapter
from src.build_index import build_index
from src.rag_engine import RAGSystem


MODES = [
    ("vector_only", "vector_only", False),
    ("lexical_only", "lexical_only", False),
    ("hybrid", "hybrid", False),
    ("hybrid_rerank", "hybrid", True),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval ablations and compute Hit@5 / MRR@10.")
    parser.add_argument("--eval-file", default="./eval_data/retrieval_eval.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rag-kb-dir", default="./processed_txt/rag_kb")
    parser.add_argument("--db-dir", default="./chroma_db")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=70)
    parser.add_argument("--auto-reindex", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-json", default="./eval_data/retrieval_eval_results.json")
    parser.add_argument("--out-csv", default="./eval_data/retrieval_eval_results.csv")
    return parser.parse_args()


def _resolve_workspace_path(raw_path: str) -> str:
    path_obj = Path(str(raw_path or "").strip())
    if path_obj.is_absolute():
        return str(path_obj)
    return str((ROOT / path_obj).resolve())


def _load_eval_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            query = str(payload.get("query", "")).strip()
            expected = payload.get("expected_sources", [])
            if not query:
                raise ValueError(f"Missing query in line {line_number}")
            if not isinstance(expected, list) or not expected:
                raise ValueError(f"Missing expected_sources in line {line_number}")
            rows.append(
                {
                    "id": str(payload.get("id", f"q{line_number}")).strip() or f"q{line_number}",
                    "query": query,
                    "expected_sources": [str(item).strip() for item in expected if str(item).strip()],
                }
            )
    if not rows:
        raise ValueError("No valid rows found in eval file.")
    return rows


def _normalize(values: List[str]) -> List[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _metrics(expected_sources: List[str], retrieved_sources: List[str]) -> Dict[str, float]:
    expected = set(_normalize(expected_sources))
    top10 = _normalize(retrieved_sources[:10])
    top5 = top10[:5]

    hit5 = 1.0 if any(source in expected for source in top5) else 0.0

    rr = 0.0
    for idx, source in enumerate(top10, start=1):
        if source in expected:
            rr = 1.0 / float(idx)
            break

    return {"hit5": hit5, "rr": rr}


def _build_vector_backend(args: argparse.Namespace) -> RAGSystem:
    rebuilt, reason = ensure_rag_index(
        rag_kb_dir=args.rag_kb_dir,
        db_dir=args.db_dir,
        rebuild_callback=lambda: build_index(
            data_folder=args.rag_kb_dir,
            db_folder=args.db_dir,
            chunk_size=int(args.chunk_size),
            chunk_overlap=int(args.chunk_overlap),
        ),
        auto_reindex=bool(args.auto_reindex),
        index_settings={
            "chunk_size": int(args.chunk_size),
            "chunk_overlap": int(args.chunk_overlap),
            "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        },
    )
    print(f"[index] {reason} (rebuilt={rebuilt})")
    return RAGSystem(db_folder=args.db_dir, k=max(int(args.k), int(args.candidate_k)))


def _build_adapter(
    mode: str,
    args: argparse.Namespace,
    vector_backend: RAGSystem | None,
) -> RetrievalAdapter:
    retrieval_mode = "hybrid" if mode.startswith("hybrid") else mode
    rag_system = vector_backend if retrieval_mode in {"vector_only", "hybrid"} else None
    return RetrievalAdapter(
        rag_system=rag_system,
        rag_kb_dir=args.rag_kb_dir,
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        retrieval_mode=retrieval_mode,
        k=int(args.k),
        candidate_k=int(args.candidate_k),
        rrf_k=60,
        ranking_profile="legal_first",
        diversify_sources=True,
        similar_score_window=0.04,
        repeat_source_penalty=0.03,
    )


def main() -> None:
    args = _parse_args()
    args.rag_kb_dir = _resolve_workspace_path(args.rag_kb_dir)
    args.db_dir = _resolve_workspace_path(args.db_dir)

    eval_rows = _load_eval_rows(args.eval_file)
    vector_backend = _build_vector_backend(args)

    per_query_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, Any]] = {}

    for label, retrieval_mode, apply_rerank in MODES:
        adapter = _build_adapter(retrieval_mode, args, vector_backend)
        mode_rows: List[Dict[str, Any]] = []
        for item in eval_rows:
            results = adapter.retrieve(query=item["query"], rerank=apply_rerank)
            top_sources = [str(row.get("source", "")).strip() for row in results[:10] if str(row.get("source", "")).strip()]
            metric = _metrics(item["expected_sources"], top_sources)
            row = {
                "mode": label,
                "id": item["id"],
                "query": item["query"],
                "expected_sources": item["expected_sources"],
                "retrieved_sources_top10": top_sources,
                "hit5": metric["hit5"],
                "rr": metric["rr"],
            }
            per_query_rows.append(row)
            mode_rows.append(row)

        n = max(1, len(mode_rows))
        hit_at_5 = sum(float(row["hit5"]) for row in mode_rows) / float(n)
        mrr_at_10 = sum(float(row["rr"]) for row in mode_rows) / float(n)
        summary[label] = {
            "queries": len(mode_rows),
            "hit_at_5": round(hit_at_5, 6),
            "mrr_at_10": round(mrr_at_10, 6),
        }

    out_json = {
        "summary": summary,
        "per_query": per_query_rows,
    }

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "id",
                "query",
                "expected_sources",
                "retrieved_sources_top10",
                "hit5",
                "rr",
            ],
        )
        writer.writeheader()
        for row in per_query_rows:
            writer.writerow(
                {
                    "mode": row["mode"],
                    "id": row["id"],
                    "query": row["query"],
                    "expected_sources": "|".join(row["expected_sources"]),
                    "retrieved_sources_top10": "|".join(row["retrieved_sources_top10"]),
                    "hit5": row["hit5"],
                    "rr": row["rr"],
                }
            )

    print("Retrieval evaluation complete")
    for mode, metrics in summary.items():
        print(f"- {mode}: Hit@5={metrics['hit_at_5']:.4f}, MRR@10={metrics['mrr_at_10']:.4f}, queries={metrics['queries']}")
    print(f"JSON: {out_json_path}")
    print(f"CSV: {out_csv_path}")


if __name__ == "__main__":
    main()
