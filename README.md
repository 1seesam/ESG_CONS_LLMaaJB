# Benchmark Workspace

Workspace zu LLM-Benchmarking, Advanced Prompting und RAG im ESG-/Immobilienkontext.

## Aktive Einstiegspunkte
- `benchmark.py`: Benchmark-only Lauf
- `evaluator.py`: LLM-as-a-Judge Bewertung
- `analyze.py`: statistische Auswertung der bewerteten Ergebnisse
- `run_pipeline.py`: End-to-End Lauf ueber alle drei Stufen
- `ingest_rag.py`: TXT-Preprocessing und Chroma-Reindex fuer den RAG-Korpus
- `tools/run_retrieval_eval.py`: Retrieval-Ablation mit Hit@5 / MRR@10

## Struktur
- `src/benchmarking/`: Kernlogik fuer Benchmark, Retrieval, Evaluation, Statistik und Reporting
- `prompts/`: aktive Prompt-Varianten fuer Zero-Shot, Advanced-Prompt, RAG und Judge
- `eval_data/`: Fragenkatalog, Master-Config und Ergebnisdateien
- `RAG_data/`: Quelldokumente fuer die Retrieval-Wissensbasis
- `processed_txt/rag_kb/`: vorverarbeitete Textbasis fuer die Indexierung
- `docs/`: Architektur- und Pipeline-Dokumentation
- `tests/`: Regressionstests fuer die produktiven Module

## Reproduzierbarer Standardlauf
```powershell
python ingest_rag.py
python run_pipeline.py --master-config ./eval_data/master_config.xlsx
```
