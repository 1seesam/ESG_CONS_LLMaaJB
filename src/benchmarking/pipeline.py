from __future__ import annotations

import os
import random
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from threading import Lock
from typing import Dict, List, Tuple

from src.build_index import build_index
from src.llm_client import get_llm
from src.rag_engine import RAGSystem

from .config_schema import BenchmarkResultRow, PathsConfig, RunConfig
from .dataset import load_questions
from .execution import calculate_cost, invoke_with_retry
from .index_manager import ensure_rag_index
from .io import append_jsonl, dataframe_from_rows, read_jsonl, write_excel, write_json, write_jsonl
from .logging_utils import config_hash, file_sha256, get_git_commit, init_run_artifacts, setup_run_logger, utc_now_iso
from .prompting import build_prompt, load_prompt_templates, prompt_templates_hash
from .pricing_provider import resolve_pricing_for_models
from .retrieval import RetrievalAdapter


DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "google/gemini-2.0-flash-exp:free": {"input": 0.00, "output": 0.00},
    # USD/token fallbacks (used only if live/cache pricing resolution fails).
    "openai/gpt-4o-mini": {"input": 1.5e-07, "output": 6.0e-07},
    "openai/gpt-4o": {"input": 2.5e-06, "output": 1.0e-05},
}


def _task_key(model_name: str, question_id: str, method: str, run_number: int) -> Tuple[str, str, str, int]:
    return (str(model_name), str(question_id), str(method), int(run_number))


def _row_task_key(row: Dict[str, object]) -> Tuple[str, str, str, int]:
    return _task_key(
        str(row.get("model", "")),
        str(row.get("question_id", "")),
        str(row.get("method", "")),
        int(row.get("run_number", 0) or 0),
    )


def _row_is_success(row: Dict[str, object]) -> bool:
    return not str(row.get("error", "")).strip()


def _canonicalize_checkpoint_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    latest_by_key: Dict[Tuple[str, str, str, int], Dict[str, object]] = {}
    for row in rows:
        try:
            key = _row_task_key(row)
        except (TypeError, ValueError):
            continue
        existing = latest_by_key.get(key)
        if existing is None or (_row_is_success(row) or not _row_is_success(existing)):
            latest_by_key[key] = dict(row)
    return list(latest_by_key.values())


def _write_compat_results_excel(results_file: str, rows: List[Dict[str, object]]) -> None:
    df = dataframe_from_rows(rows)
    if df.empty:
        return
    compat_df = df.rename(
        columns={
            "run_number": "Run_ID",
            "model": "Model",
            "method": "Method",
            "question_id": "ID",
            "question": "Frage",
            "answer_type": "Antworttyp",
            "expected_trap_behavior": "Expected_Trap_Behavior",
            "llm_answer": "KI_Antwort",
            "reference_answer": "Referenzantwort",
            "atomic_facts": "Atomic_Facts",
            "document_name": "Dokument_Name",
            "input_tokens": "Input_Tokens",
            "cost_usd": "Cost_USD",
            "duration_sec": "Dauer_sek",
        }
    )
    write_excel(results_file, compat_df)


def _render_progress(
    done: int,
    total: int,
    started_at: float,
    label: str,
    width: int = 28,
) -> str:
    if total <= 0:
        return f"[{'-' * width}]   0.0% | {label}"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - started_at, 0.001)
    avg = elapsed / max(done, 1)
    remaining = max(total - done, 0)
    eta = int(avg * remaining)
    return f"[{bar}] {ratio*100:5.1f}% | {done}/{total} | ETA {eta:>4}s | {label}"


def _load_eval_document(eval_docs_dir: str, relevant_docs: str) -> (str, str):
    raw_doc_name = str(relevant_docs or "").strip()
    if not raw_doc_name or raw_doc_name.lower() == "nan":
        return "", ""
    base_name = os.path.splitext(raw_doc_name)[0]
    txt_filename = base_name + ".txt"
    txt_path = os.path.join(eval_docs_dir, txt_filename)
    if not os.path.exists(txt_path):
        return "", txt_filename
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read(), txt_filename


def run_benchmark(config: RunConfig, paths: PathsConfig | None = None, run_uuid: str | None = None):
    paths = paths or PathsConfig()
    random.seed(config.seed)

    artifacts = init_run_artifacts(paths.runs_root, run_uuid=run_uuid)
    logger = setup_run_logger("benchmark-run", artifacts["run_log"])
    print(f"Benchmark run_uuid: {artifacts['run_uuid']}")
    print(f"Benchmark run_dir: {artifacts['run_dir']}")
    logger.info("Benchmark run_uuid: %s", artifacts["run_uuid"])
    logger.info("Benchmark run_dir: %s", artifacts["run_dir"])

    if not os.path.exists(config.question_file):
        raise FileNotFoundError(f"Input file not found: {config.question_file}")

    questions = load_questions(config.question_file)

    run_config_snapshot = config.to_dict()
    run_config_snapshot["question_file_sha256"] = file_sha256(config.question_file)
    run_config_snapshot["git_commit"] = get_git_commit(os.getcwd())
    pricing_map, pricing_details = resolve_pricing_for_models(
        models=config.models,
        default_pricing=DEFAULT_PRICING,
        cache_path="./eval_data/pricing_cache.json",
    )
    write_json(
        artifacts["pricing_snapshot"],
        {
            "models": config.models,
            "pricing_map_used": pricing_map,
            "details": pricing_details,
        },
    )
    templates = load_prompt_templates(
        {
            "Zero-Shot": config.prompt_zero_shot_path,
            "Advanced-Prompt": config.prompt_advanced_path,
            "RAG": config.prompt_rag_path,
        }
    )
    template_paths = {
        "Zero-Shot": config.prompt_zero_shot_path,
        "Advanced-Prompt": config.prompt_advanced_path,
        "RAG": config.prompt_rag_path,
    }
    for method_name, path in template_paths.items():
        path_text = str(path or "").strip()
        if path_text and os.path.exists(path_text):
            msg = f"Prompt template [{method_name}]: CUSTOM ({path_text})"
        elif path_text:
            msg = f"Prompt template [{method_name}]: DEFAULT (missing custom path: {path_text})"
        else:
            msg = f"Prompt template [{method_name}]: DEFAULT (no custom path)"
        print(msg)
        logger.info(msg)
    run_config_snapshot["prompt_templates_hash"] = prompt_templates_hash(templates)
    run_config_snapshot["prompt_template_paths"] = {
        "Zero-Shot": config.prompt_zero_shot_path,
        "Advanced-Prompt": config.prompt_advanced_path,
        "RAG": config.prompt_rag_path,
    }
    pricing_source = str(pricing_details.get("source", "unknown"))
    print(f"Pricing source: {pricing_source} (models: {len(config.models)})")
    logger.info("Pricing source: %s", pricing_source)
    if pricing_details.get("fetch_error"):
        logger.warning("Pricing fetch issue: %s", pricing_details.get("fetch_error"))
    run_config_snapshot["pricing_snapshot"] = artifacts["pricing_snapshot"]
    run_config_snapshot["pricing_details"] = pricing_details
    run_hash = config_hash(run_config_snapshot)
    write_json(artifacts["config_snapshot"], run_config_snapshot)

    rag_system = None
    if config.use_rag and "RAG" in config.methods:
        try:
            index_settings = {
                "chunk_size": int(config.rag_chunk_size),
                "chunk_overlap": int(config.rag_chunk_overlap),
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            }
            rebuilt, reason = ensure_rag_index(
                rag_kb_dir=paths.rag_kb_dir,
                db_dir="./chroma_db",
                rebuild_callback=lambda: build_index(
                    data_folder=paths.rag_kb_dir,
                    db_folder="./chroma_db",
                    chunk_size=int(config.rag_chunk_size),
                    chunk_overlap=int(config.rag_chunk_overlap),
                ),
                auto_reindex=config.auto_reindex,
                index_settings=index_settings,
            )
            logger.info("RAG index check: %s (rebuilt=%s)", reason, rebuilt)
            rag_system = RAGSystem()
        except Exception as exc:  # pragma: no cover - external DB state
            logger.warning("RAG disabled due to init error: %s", exc)
            rag_system = None
    if rag_system is not None and config.use_rag and "RAG" in config.methods:
        retriever = RetrievalAdapter(
            rag_system,
            rag_kb_dir=paths.rag_kb_dir,
            chunk_size=int(config.rag_chunk_size),
            chunk_overlap=int(config.rag_chunk_overlap),
            retrieval_mode="hybrid",
            k=5,
            candidate_k=50,
            rrf_k=60,
            ranking_profile="legal_first",
            diversify_sources=True,
            similar_score_window=0.04,
            repeat_source_penalty=0.03,
        )
    else:
        retriever = RetrievalAdapter(rag_system, retrieval_mode="vector_only", k=5, candidate_k=50)

    def retrieve_context_with_fallback(question: str, doc_name: str, has_doc_reference: bool) -> Tuple[str, List[str]]:
        filename_filter = doc_name if has_doc_reference else None
        rag_ctx, sources = retriever.get_context(question, filename_filter=filename_filter)
        if filename_filter and not rag_ctx.strip():
            logger.info(
                "RAG filtered retrieval empty for source=%s; falling back to unfiltered retrieval.",
                filename_filter,
            )
            rag_ctx, sources = retriever.get_context(question, filename_filter=None)
        return rag_ctx, sources

    # Build concrete task list first, so progress can be exact.
    base_tasks = []
    for item in questions:
        doc_text, doc_name = _load_eval_document(paths.eval_docs_dir, item.relevant_docs)
        for method in config.methods:
            base_tasks.append((item, method, "", doc_name, bool(doc_name)))

    total_steps = len(base_tasks) * len(config.models) * config.num_runs
    started_at = time.time()
    planned_task_keys = {
        _task_key(model_name, item.id, method, run_idx + 1)
        for model_name in config.models
        for item, method, _, _, _ in base_tasks
        for run_idx in range(config.num_runs)
    }
    checkpoint_rows = _canonicalize_checkpoint_rows(read_jsonl(artifacts["raw_results_jsonl"]))
    completed_task_keys = {
        _row_task_key(row)
        for row in checkpoint_rows
        if _row_is_success(row) and _row_task_key(row) in planned_task_keys
    }
    if checkpoint_rows:
        write_jsonl(artifacts["raw_results_jsonl"], checkpoint_rows)
        logger.info(
            "Resume checkpoint loaded: %s rows, %s completed tasks.",
            len(checkpoint_rows),
            len(completed_task_keys),
        )
        print(
            f"Resume checkpoint gefunden: {len(completed_task_keys)}/{total_steps} Tasks bereits erfolgreich."
        )
    done_steps = len(completed_task_keys)
    print(f"Benchmark progress: {done_steps}/{total_steps} steps")
    parallel_workers = max(1, int(getattr(config, "parallel_workers", 1) or 1))
    if parallel_workers > 1:
        print(f"Benchmark parallel_workers: {parallel_workers}")

    rows_by_key = {
        _row_task_key(row): dict(row)
        for row in checkpoint_rows
        if _row_task_key(row) in planned_task_keys
    }
    consecutive_api_fails = 0
    append_lock = Lock()
    for model_name in config.models:
        pending_for_model = any(
            _task_key(model_name, item.id, method, run_idx + 1) not in completed_task_keys
            for item, method, _, _, _ in base_tasks
            for run_idx in range(config.num_runs)
        )
        if not pending_for_model:
            logger.info("Skipping model %s: all planned tasks already completed.", model_name)
            continue
        logger.info("Loading model: %s", model_name)
        model_temperature = config.model_temperatures.get(model_name, config.temperature)
        try:
            llm = get_llm(model_name=model_name, temperature=model_temperature)
        except Exception as exc:
            logger.error("Model load failed for %s: %s", model_name, exc)
            continue

        def execute_task(task_idx: int, item, method: str, input_doc: str, doc_name: str, has_doc_reference: bool, run_idx: int):
            rag_ctx = ""
            sources: List[str] = []
            if method == "RAG" and rag_system is not None:
                rag_ctx, sources = retrieve_context_with_fallback(
                    item.question,
                    doc_name,
                    has_doc_reference,
                )
                if len(rag_ctx) > config.max_rag_context_chars:
                    rag_ctx = rag_ctx[: config.max_rag_context_chars]

            prompt = build_prompt(
                method=method,
                question=item.question,
                input_doc_text=input_doc,
                rag_context=rag_ctx,
                answer_type=item.answer_type,
                max_words=(
                    str(config.default_max_words)
                    if config.default_max_words is not None
                    else item.max_words
                ),
                atomic_facts=item.atomic_facts,
                prompt_templates=templates,
            )
            if len(prompt) > config.max_prompt_chars:
                prompt = prompt[: config.max_prompt_chars] + "\n\n[Prompt gekuerzt wegen max_prompt_chars]"

            result = invoke_with_retry(
                llm=llm,
                prompt=prompt,
                max_retries=config.max_retries,
                throttle_seconds=config.throttle_seconds,
            )
            cost = calculate_cost(
                model_name,
                result["input_tokens"],
                result["output_tokens"],
                pricing_map,
            )
            task_key = _task_key(model_name, item.id, method, run_idx + 1)
            row = asdict(
                BenchmarkResultRow(
                    run_uuid=artifacts["run_uuid"],
                    timestamp_utc=utc_now_iso(),
                    config_hash=run_hash,
                    run_number=run_idx + 1,
                    model=model_name,
                    method=method,
                    question_id=item.id,
                    question=item.question,
                    answer_type=item.answer_type,
                    expected_trap_behavior=item.expected_trap_behavior,
                    llm_answer=result["content"],
                    reference_answer=item.reference_answer,
                    atomic_facts=json.dumps(item.atomic_facts, ensure_ascii=False),
                    document_name=doc_name if has_doc_reference else "n/a",
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    cost_usd=cost,
                    duration_sec=result["duration_sec"],
                    retrieval_sources="|".join(sorted(set(sources))),
                    error=result["error"],
                )
            )
            with append_lock:
                append_jsonl(artifacts["raw_results_jsonl"], [row])
            return task_idx, task_key, row, result, len(prompt), item, method, run_idx

        if parallel_workers > 1:
            pending_tasks = []
            for task_idx, (item, method, input_doc, doc_name, has_doc_reference) in enumerate(base_tasks):
                for run_idx in range(config.num_runs):
                    task_key = _task_key(model_name, item.id, method, run_idx + 1)
                    if task_key not in completed_task_keys:
                        pending_tasks.append((task_idx, item, method, input_doc, doc_name, has_doc_reference, run_idx))

            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [
                    executor.submit(execute_task, task_idx, item, method, input_doc, doc_name, has_doc_reference, run_idx)
                    for task_idx, item, method, input_doc, doc_name, has_doc_reference, run_idx in pending_tasks
                ]
                for future in as_completed(futures):
                    task_idx, task_key, row, result, prompt_len, item, method, run_idx = future.result()
                    rows_by_key[task_key] = row
                    if result["error"]:
                        consecutive_api_fails += 1
                        error_msg = (
                            f"API FAIL {consecutive_api_fails}/{config.max_consecutive_api_fails} | "
                            f"model={model_name} id={item.id} method={method} run={run_idx+1} | {result['error']}"
                        )
                        print("\n" + error_msg, flush=True)
                        logger.warning(error_msg)
                        if consecutive_api_fails > config.max_consecutive_api_fails:
                            canonical_rows = [rows_by_key[key] for key in rows_by_key]
                            write_jsonl(artifacts["raw_results_jsonl"], canonical_rows)
                            _write_compat_results_excel(config.results_file, canonical_rows)
                            raise RuntimeError(
                                "Stopped benchmark: more than "
                                f"{config.max_consecutive_api_fails} API failures. "
                                f"Resume by rerunning with run_uuid={artifacts['run_uuid']}."
                            )
                    else:
                        consecutive_api_fails = 0
                        completed_task_keys.add(task_key)

                    if result["duration_sec"] >= config.slow_call_seconds:
                        slow_msg = (
                            f"SLOW CALL {result['duration_sec']:.2f}s | "
                            f"model={model_name} id={item.id} method={method} run={run_idx+1} "
                            f"prompt_chars={prompt_len}"
                        )
                        print("\n" + slow_msg, flush=True)
                        logger.warning(slow_msg)

                    done_steps += 1
                    label = f"{model_name} | Q{item.id} | {method} | run {run_idx + 1}/{config.num_runs}"
                    print("\r" + _render_progress(done_steps, total_steps, started_at, label), end="", flush=True)

            print()
            continue

        for task_idx, (item, method, input_doc, doc_name, has_doc_reference) in enumerate(base_tasks):
            for run_idx in range(config.num_runs):
                task_key = _task_key(model_name, item.id, method, run_idx + 1)
                if task_key in completed_task_keys:
                    continue
                label = f"{model_name} | Q{item.id} | {method} | run {run_idx + 1}/{config.num_runs}"
                print("\r" + _render_progress(done_steps, total_steps, started_at, label), end="", flush=True)
                logger.info(
                    "Q task %s/%s | id=%s | model=%s | method=%s | run=%s",
                    task_idx + 1,
                    len(base_tasks),
                    item.id,
                    model_name,
                    method,
                    run_idx + 1,
                )

                rag_ctx = ""
                sources: List[str] = []
                if method == "RAG" and rag_system is not None:
                    rag_ctx, sources = retrieve_context_with_fallback(
                        item.question,
                        doc_name,
                        has_doc_reference,
                    )
                    if len(rag_ctx) > config.max_rag_context_chars:
                        rag_ctx = rag_ctx[: config.max_rag_context_chars]

                prompt = build_prompt(
                    method=method,
                    question=item.question,
                    input_doc_text=input_doc,
                    rag_context=rag_ctx,
                    answer_type=item.answer_type,
                    max_words=(
                        str(config.default_max_words)
                        if config.default_max_words is not None
                        else item.max_words
                    ),
                    atomic_facts=item.atomic_facts,
                    prompt_templates=templates,
                )
                if len(prompt) > config.max_prompt_chars:
                    prompt = prompt[: config.max_prompt_chars] + "\n\n[Prompt gekuerzt wegen max_prompt_chars]"

                result = invoke_with_retry(
                    llm=llm,
                    prompt=prompt,
                    max_retries=config.max_retries,
                    throttle_seconds=config.throttle_seconds,
                )
                cost = calculate_cost(
                    model_name,
                    result["input_tokens"],
                    result["output_tokens"],
                    pricing_map,
                )
                row = asdict(
                    BenchmarkResultRow(
                        run_uuid=artifacts["run_uuid"],
                        timestamp_utc=utc_now_iso(),
                        config_hash=run_hash,
                        run_number=run_idx + 1,
                        model=model_name,
                        method=method,
                        question_id=item.id,
                        question=item.question,
                        answer_type=item.answer_type,
                        expected_trap_behavior=item.expected_trap_behavior,
                        llm_answer=result["content"],
                        reference_answer=item.reference_answer,
                        atomic_facts=json.dumps(item.atomic_facts, ensure_ascii=False),
                        document_name=doc_name if has_doc_reference else "n/a",
                        input_tokens=result["input_tokens"],
                        output_tokens=result["output_tokens"],
                        cost_usd=cost,
                        duration_sec=result["duration_sec"],
                        retrieval_sources="|".join(sorted(set(sources))),
                        error=result["error"],
                    )
                )
                rows_by_key[task_key] = row
                append_jsonl(artifacts["raw_results_jsonl"], [row])

                if result["error"]:
                    consecutive_api_fails += 1
                    error_msg = (
                        f"API FAIL {consecutive_api_fails}/{config.max_consecutive_api_fails} | "
                        f"model={model_name} id={item.id} method={method} run={run_idx+1} | {result['error']}"
                    )
                    print("\n" + error_msg, flush=True)
                    logger.warning(error_msg)
                    if consecutive_api_fails > config.max_consecutive_api_fails:
                        canonical_rows = [rows_by_key[key] for key in rows_by_key]
                        write_jsonl(artifacts["raw_results_jsonl"], canonical_rows)
                        _write_compat_results_excel(config.results_file, canonical_rows)
                        raise RuntimeError(
                            "Stopped benchmark: more than "
                            f"{config.max_consecutive_api_fails} consecutive API failures. "
                            f"Resume by rerunning with run_uuid={artifacts['run_uuid']}."
                        )
                else:
                    consecutive_api_fails = 0
                    completed_task_keys.add(task_key)

                if result["duration_sec"] >= config.slow_call_seconds:
                    slow_msg = (
                        f"SLOW CALL {result['duration_sec']:.2f}s | "
                        f"model={model_name} id={item.id} method={method} run={run_idx+1} "
                        f"prompt_chars={len(prompt)}"
                    )
                    print("\n" + slow_msg, flush=True)
                    logger.warning(slow_msg)

                done_steps += 1

        # line break after each model
        print()

    if total_steps:
        print("\r" + _render_progress(done_steps, total_steps, started_at, "completed"), flush=True)

    rows = [rows_by_key[key] for key in rows_by_key]
    raw_jsonl = artifacts["raw_results_jsonl"]
    write_jsonl(raw_jsonl, rows)
    _write_compat_results_excel(config.results_file, rows)
    df = dataframe_from_rows(rows)
    return df, artifacts
