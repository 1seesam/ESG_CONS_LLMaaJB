# Current Architecture Audit

| Component | Responsibility | Critical points |
|---|---|---|
| `benchmark.py` | Benchmark-only entrypoint with optional Excel->JSON sync via master config. | Keeps one productive benchmark CLI and removes duplicate wrappers. |
| `run_pipeline.py` | End-to-end orchestration for benchmark, evaluation and analysis. | Reuses one shared run context and one master config. |
| `ingest_rag.py` | Preprocesses the active RAG corpus and rebuilds the Chroma index. | Keeps ingestion separate from benchmark/evaluation logic. |
| `src/benchmarking/*` | Active benchmark, retrieval, scoring, statistics and orchestration modules. | Public method set is reduced to `Zero-Shot`, `Advanced-Prompt`, `RAG`. |
| `src/build_index.py` + `src/rag_engine.py` | RAG indexing and vector backend for retrieval. | Chroma index is auto-rebuilt from `processed_txt/rag_kb` when inputs change. |
| `tools/run_retrieval_eval.py` | Retrieval ablation runner on the integrated retrieval stack. | Uses `RetrievalAdapter` directly instead of prototype helpers. |

## Refactor target
- `src/benchmarking/` hosts the thesis core for benchmark, prompting, retrieval, evaluation and reporting.
- Supported CLIs: `benchmark.py`, `evaluator.py`, `analyze.py`, `run_pipeline.py`, `ingest_rag.py`.
- Removed scratch/legacy surface: ad-hoc scripts such as `rag.py`, `test.py`, `test copy.py` and duplicate visualization variants.

