from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STOPWORDS = {
    "der",
    "die",
    "das",
    "und",
    "oder",
    "ein",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    "zu",
    "im",
    "in",
    "mit",
    "auf",
    "fuer",
    "von",
    "ist",
    "sind",
    "als",
    "bei",
    "nach",
    "ab",
    "an",
    "dem",
    "den",
    "des",
    "am",
    "it",
    "the",
    "and",
    "or",
    "to",
    "of",
}

PROFILE_WEIGHTS = {
    "neutral": {"law": 0.0, "standard": 0.0, "technical": 0.0, "web": 0.0},
    "legal_first": {"law": 0.18, "standard": 0.08, "technical": 0.0, "web": -0.08},
}

MARKER_PATTERNS = [
    re.compile(r"^=== DOKUMENT START:.*$"),
    re.compile(r"^=== DOKUMENT ENDE ===$"),
    re.compile(r"^--- PDF SEITE \d+ ---$"),
]


class RetrievalAdapter:
    def __init__(
        self,
        rag_system,
        rag_kb_dir: str = "./processed_txt/rag_kb",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        retrieval_mode: str = "hybrid",
        k: int = 5,
        candidate_k: int = 50,
        rrf_k: int = 60,
        ranking_profile: str = "legal_first",
        diversify_sources: bool = True,
        similar_score_window: float = 0.04,
        repeat_source_penalty: float = 0.03,
    ):
        self.rag_system = rag_system
        self.rag_kb_dir = str(rag_kb_dir)
        self.chunk_size = max(300, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.retrieval_mode = str(retrieval_mode or "hybrid")
        if self.retrieval_mode not in {"vector_only", "lexical_only", "hybrid"}:
            self.retrieval_mode = "hybrid"
        self.k = max(1, int(k))
        self.candidate_k = max(50, int(candidate_k), self.k)
        self.rrf_k = max(1, int(rrf_k))
        self.ranking_profile = str(ranking_profile or "legal_first")
        self.diversify_sources = bool(diversify_sources)
        self.similar_score_window = max(0.0, float(similar_score_window))
        self.repeat_source_penalty = max(0.0, float(repeat_source_penalty))

        self._lexical_index: Optional[Dict[str, Any]] = None
        if self.retrieval_mode in {"lexical_only", "hybrid"}:
            self._lexical_index = self._build_bm25_index(self._load_chunk_rows())

    @staticmethod
    def _normalize_ws(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9\-]+", str(text or "").lower())
        return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def _split_text(self, text: str) -> List[str]:
        cleaned = self._clean_text_for_chunking(text)
        if not cleaned:
            return []

        lines = cleaned.splitlines()
        sections: List[str] = []
        current: List[str] = []

        for line in lines:
            if self._is_heading_line(line) and current:
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
            if len(section) <= self.chunk_size:
                chunks.append(self._normalize_ws(section))
                continue
            for piece in self._fixed_size_split(section, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap):
                chunks.append(self._normalize_ws(piece))

        return [chunk for chunk in chunks if chunk]

    def _load_chunk_rows(self) -> List[Dict[str, Any]]:
        base = Path(self.rag_kb_dir)
        txt_files = sorted(path for path in base.glob("*.txt") if path.is_file())
        rows: List[Dict[str, Any]] = []

        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8")
            parts = self._split_text(text)
            for idx, part in enumerate(parts, start=1):
                rows.append(
                    {
                        "chunk_id": f"{txt_file.name}::chunk_{idx:04d}",
                        "source": txt_file.name,
                        "chunk": self._normalize_ws(part),
                    }
                )
        return rows

    def _build_bm25_index(self, rows: List[Dict[str, Any]], k1: float = 1.2, b: float = 0.75) -> Dict[str, Any]:
        tf_list: List[Counter[str]] = []
        doc_lengths: List[int] = []
        df = Counter()

        for row in rows:
            tokens = self._tokenize(row["chunk"])
            tf = Counter(tokens)
            tf_list.append(tf)
            doc_lengths.append(len(tokens))
            for tok in tf:
                df[tok] += 1

        n_docs = max(1, len(rows))
        avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 1.0
        idf: Dict[str, float] = {}
        for tok, freq in df.items():
            idf[tok] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

        return {
            "rows": rows,
            "tf_list": tf_list,
            "doc_lengths": doc_lengths,
            "avgdl": avgdl,
            "idf": idf,
            "k1": float(k1),
            "b": float(b),
        }

    def _bm25_search(self, index: Dict[str, Any], query: str, k: int, source: str = "") -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        query_tf = Counter(query_tokens)

        rows: List[Dict[str, Any]] = index["rows"]
        tf_list: List[Counter[str]] = index["tf_list"]
        doc_lengths: List[int] = index["doc_lengths"]
        avgdl = float(index["avgdl"] or 1.0)
        idf: Dict[str, float] = index["idf"]
        k1 = float(index["k1"])
        b = float(index["b"])

        source_filter = str(source or "").strip()
        scored: List[Tuple[float, int]] = []

        for idx, row in enumerate(rows):
            if source_filter and row["source"] != source_filter:
                continue
            tf = tf_list[idx]
            dl = max(1, int(doc_lengths[idx]))
            score = 0.0

            for tok, qf in query_tf.items():
                tfv = tf.get(tok, 0)
                if tfv <= 0:
                    continue
                token_idf = idf.get(tok, 0.0)
                denom = tfv + k1 * (1.0 - b + b * (dl / avgdl))
                score += float(qf) * token_idf * ((tfv * (k1 + 1.0)) / max(1e-9, denom))

            if score > 0.0:
                scored.append((score, idx))

        if not scored:
            for idx, row in enumerate(rows):
                if source_filter and row["source"] != source_filter:
                    continue
                scored.append((0.0, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(scored[: max(1, int(k))], start=1):
            row = rows[idx]
            out.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "source": row["source"],
                    "chunk_id": row["chunk_id"],
                    "chunk": row["chunk"],
                }
            )
        return out

    @staticmethod
    def _search_chroma(vectorstore, query: str, k: int, source: str = "") -> List[Dict[str, Any]]:
        search_kwargs: Dict[str, Any] = {"k": max(1, int(k))}
        source_filter = str(source or "").strip()
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}

        rows: List[Dict[str, Any]] = []
        with_score = getattr(vectorstore, "similarity_search_with_score", None)

        if callable(with_score):
            for rank, (doc, score) in enumerate(with_score(query, **search_kwargs), start=1):
                src = str(doc.metadata.get("source", "unknown"))
                chunk_id = str(doc.metadata.get("chunk_id", f"{src}::chunk_unknown"))
                rows.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "source": src,
                        "chunk_id": chunk_id,
                        "chunk": str(doc.page_content or ""),
                    }
                )
            return rows

        docs = vectorstore.similarity_search(query, **search_kwargs)
        for rank, doc in enumerate(docs, start=1):
            src = str(doc.metadata.get("source", "unknown"))
            chunk_id = str(doc.metadata.get("chunk_id", f"{src}::chunk_unknown"))
            rows.append(
                {
                    "rank": rank,
                    "score": None,
                    "source": src,
                    "chunk_id": chunk_id,
                    "chunk": str(doc.page_content or ""),
                }
            )
        return rows

    def _rrf_fuse(self, lexical_rows: List[Dict[str, Any]], vector_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        def consider(row: Dict[str, Any], rank: int, engine: str) -> None:
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id:
                return

            contribution = 1.0 / (float(self.rrf_k) + float(rank))
            existing = merged.get(chunk_id)
            if existing is None:
                payload = dict(row)
                payload["fusion_score"] = contribution
                payload["retrieval_engine"] = engine
                merged[chunk_id] = payload
                return

            existing["fusion_score"] = float(existing.get("fusion_score", 0.0)) + contribution
            existing_engine = "chroma" if engine == "vector" else "lexical"
            current_engine = "chroma" if str(existing.get("retrieval_engine")) == "vector" else "lexical"
            candidate_rel = self._score_to_relevance(row.get("score"), existing_engine)
            current_rel = self._score_to_relevance(existing.get("score"), current_engine)
            if candidate_rel > current_rel:
                for key in ("score", "source", "chunk", "retrieval_engine"):
                    existing[key] = row.get(key, existing.get(key))

        for rank, row in enumerate(lexical_rows, start=1):
            consider(row=row, rank=rank, engine="lexical")
        for rank, row in enumerate(vector_rows, start=1):
            consider(row=row, rank=rank, engine="vector")

        fused = list(merged.values())
        fused.sort(key=lambda x: float(x.get("fusion_score", 0.0)), reverse=True)
        for idx, row in enumerate(fused, start=1):
            row["rank"] = idx
        return fused

    @staticmethod
    def _classify_source(source: str) -> str:
        s = str(source or "").lower()
        law_markers = (
            "_t.txt",
            "gesetz",
            "verordnung",
            "richtlinie",
            "directive",
            "csrd",
            "sfdr",
            "epbd",
            "epra",
            "eu-tax",
            "geg",
            "geig",
            "bafa",
        )
        standard_markers = ("din", "iso", "vdi", "norm")
        web_markers = ("wikipedia", "baunetz", "wissen", "blog", "magazin")

        if any(marker in s for marker in law_markers):
            return "law"
        if any(marker in s for marker in standard_markers):
            return "standard"
        if any(marker in s for marker in web_markers):
            return "web"
        return "technical"

    @staticmethod
    def _score_to_relevance(score: Any, engine: str) -> float:
        if score is None:
            return 0.0
        try:
            value = float(score)
        except Exception:
            return 0.0
        if engine == "chroma":
            return 1.0 / (1.0 + max(0.0, value))
        return max(0.0, value)

    def _rerank_results(self, rows: List[Dict[str, Any]], engine: str) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        weights = PROFILE_WEIGHTS.get(self.ranking_profile, PROFILE_WEIGHTS["neutral"])

        working: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            source_type = self._classify_source(item.get("source", ""))
            base_rel = self._score_to_relevance(item.get("score"), engine=engine)
            item["source_type"] = source_type
            item["base_relevance"] = float(base_rel)
            item["source_weight"] = float(weights.get(source_type, 0.0))
            item["rerank_score"] = item["base_relevance"] + item["source_weight"]
            working.append(item)

        if engine != "chroma":
            max_rel = max((float(x.get("base_relevance", 0.0)) for x in working), default=1.0) or 1.0
            for item in working:
                item["base_relevance"] = float(item["base_relevance"]) / max_rel
                item["rerank_score"] = item["base_relevance"] + float(item["source_weight"])

        selected: List[Dict[str, Any]] = []
        remaining = list(working)
        source_counts: Dict[str, int] = {}

        while remaining and len(selected) < self.k:
            scored: List[Dict[str, Any]] = []
            for item in remaining:
                src = str(item.get("source", ""))
                penalty = self.repeat_source_penalty * source_counts.get(src, 0) if self.diversify_sources else 0.0
                cand = dict(item)
                cand["final_score"] = float(item.get("rerank_score", 0.0)) - penalty
                scored.append(cand)

            scored.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
            chosen = scored[0]

            if self.diversify_sources and selected and len(scored) > 1:
                seen_sources = {str(x.get("source", "")) for x in selected}
                if str(chosen.get("source", "")) in seen_sources:
                    top_score = float(chosen.get("final_score", 0.0))
                    for alt in scored[1:6]:
                        alt_src = str(alt.get("source", ""))
                        if alt_src in seen_sources:
                            continue
                        if (top_score - float(alt.get("final_score", 0.0))) <= self.similar_score_window:
                            chosen = alt
                            break

            selected.append(chosen)
            src = str(chosen.get("source", ""))
            source_counts[src] = source_counts.get(src, 0) + 1
            remaining = [row for row in remaining if row.get("chunk_id") != chosen.get("chunk_id")]

        for idx, row in enumerate(selected, start=1):
            row["rank"] = idx
        return selected

    def _retrieve_rows(
        self,
        query: str,
        filename_filter: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        source_filter = str(filename_filter or "").strip()
        mode = self.retrieval_mode
        candidate_k = self.candidate_k

        lexical_rows: List[Dict[str, Any]] = []
        vector_rows: List[Dict[str, Any]] = []

        if mode in {"lexical_only", "hybrid"} and self._lexical_index is not None:
            lexical_rows = self._bm25_search(self._lexical_index, query=query, k=candidate_k, source=source_filter)

        if mode in {"vector_only", "hybrid"} and self.rag_system is not None:
            vector_rows = self._search_chroma(
                self.rag_system.vectorstore,
                query=query,
                k=candidate_k,
                source=source_filter,
            )

        if mode == "lexical_only":
            if not rerank:
                return lexical_rows
            return self._rerank_results(lexical_rows, engine="lexical")
        if mode == "vector_only":
            if not rerank:
                return vector_rows
            return self._rerank_results(vector_rows, engine="chroma")

        if lexical_rows and vector_rows:
            fused = self._rrf_fuse(lexical_rows=lexical_rows, vector_rows=vector_rows)
            if not rerank:
                return fused
            prepared = [dict(row, score=float(row.get("fusion_score", 0.0))) for row in fused]
            return self._rerank_results(prepared, engine="hybrid")

        if vector_rows:
            if not rerank:
                return vector_rows
            return self._rerank_results(vector_rows, engine="chroma")
        if not rerank:
            return lexical_rows
        return self._rerank_results(lexical_rows, engine="lexical")

    def get_context(self, query: str, filename_filter: Optional[str] = None) -> Tuple[str, List[str]]:
        rows = self.retrieve(query=query, filename_filter=filename_filter, rerank=True)
        top_rows = rows[: self.k]

        sources: List[str] = []
        chunks: List[str] = []
        for row in top_rows:
            source = str(row.get("source", "")).strip()
            chunk = str(row.get("chunk", "")).strip()
            if source:
                sources.append(source)
            if chunk:
                chunks.append(chunk)

        context = "\n\n---\n\n".join(chunks)
        return context, sources

    def retrieve(
        self,
        query: str,
        filename_filter: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._retrieve_rows(query=query, filename_filter=filename_filter, rerank=rerank)
