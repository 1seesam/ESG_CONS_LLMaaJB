from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Dict, Tuple

import pandas as pd


from src.llm_client import get_llm

from .config_schema import EvaluationRow
from .execution import calculate_cost, invoke_with_retry
from .io import read_excel, write_excel, write_json, write_jsonl
from .logging_utils import init_run_artifacts
from .pricing_provider import resolve_pricing_for_models
from .scoring import (
    build_atomic_facts_judge_prompt,
    derive_likert_from_claim_counts,
    extract_confidence_mean,
    is_trap_question,
    parse_atomic_facts_from_row,
    parse_claim_labels_json,
    parse_judge_json,
    parse_predicted_claim_labels_json,
    sanitize_judge_text,
)


JUDGE_ISSUE_MARKERS = (
    "API Error",
    "Judge JSON parse failed",
    "Judge returned empty response",
    "Benchmark API Error",
)


def _to_float(value, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out


def _to_int(value, default: int = 0) -> int:
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _judge_reason_has_issue(reason: object) -> bool:
    text = str(reason or "")
    return any(marker.lower() in text.lower() for marker in JUDGE_ISSUE_MARKERS)


def _evaluation_key_columns(df: pd.DataFrame, existing: pd.DataFrame | None) -> list[str]:
    preferred = ["run_uuid", "config_hash", "Run_ID", "Model", "Method", "ID"]
    key_cols = [col for col in preferred if col in df.columns and (existing is None or col in existing.columns)]
    return key_cols if {"Run_ID", "Model", "Method", "ID"}.issubset(set(key_cols)) else []


def _row_key(row, key_cols: list[str]) -> tuple[str, ...]:
    return tuple(_cell_text(row.get(col, "")) for col in key_cols)


def _source_fields_match(current, cached) -> bool:
    source_columns = [
        "Frage",
        "Antworttyp",
        "KI_Antwort",
        "Referenzantwort",
        "Atomic_Facts",
        "Expected_Trap_Behavior",
        "error",
    ]
    for col in source_columns:
        if col in cached.index and _cell_text(current.get(col, "")) != _cell_text(cached.get(col, "")):
            return False
    return True


def _judge_prompt_hash(judge_prompt_template: str | None) -> str:
    sample_fact = "__FACT__"
    payload = "\n---non_trap---\n".join(
        [
            build_atomic_facts_judge_prompt(
                question="__QUESTION__",
                generated="__ANSWER__",
                atomic_facts=[sample_fact],
                judge_prompt_template=judge_prompt_template,
                trap_mode=False,
            ),
            build_atomic_facts_judge_prompt(
                question="__TRAP_QUESTION__",
                generated="__TRAP_ANSWER__",
                atomic_facts=[sample_fact],
                judge_prompt_template=judge_prompt_template,
                trap_mode=True,
                expected_trap_behavior="__EXPECTED_TRAP_BEHAVIOR__",
            ),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_evaluation_is_reusable(
    row,
    judge_model: str,
    judge_prompt_version: str,
    judge_prompt_hash: str,
) -> bool:
    if _judge_reason_has_issue(row.get("Judge_Begruendung", "")):
        return False
    if "judge_model" in row.index and _cell_text(row.get("judge_model", "")) != _cell_text(judge_model):
        return False
    if "judge_prompt_version" in row.index and _cell_text(row.get("judge_prompt_version", "")) != _cell_text(
        judge_prompt_version
    ):
        return False
    if _cell_text(row.get("judge_prompt_hash", "")) != _cell_text(judge_prompt_hash):
        return False
    score = _to_float(row.get("Likert_Score", math.nan))
    return not math.isnan(score)


def _load_evaluation_cache(
    output_file: str,
    input_df: pd.DataFrame,
    judge_model: str,
    judge_prompt_version: str,
    judge_prompt_hash: str,
) -> tuple[dict[tuple[str, ...], pd.Series], list[str]]:
    if not os.path.exists(output_file):
        return {}, []
    try:
        existing = read_excel(output_file)
    except Exception:
        return {}, []
    key_cols = _evaluation_key_columns(input_df, existing)
    if not key_cols:
        return {}, []
    cache: dict[tuple[str, ...], pd.Series] = {}
    for _, row in existing.iterrows():
        if not _cached_evaluation_is_reusable(row, judge_model, judge_prompt_version, judge_prompt_hash):
            continue
        cache[_row_key(row, key_cols)] = row
    return cache, key_cols


def _render_eval_progress(done: int, total: int, started_at: float, width: int = 28) -> str:
    if total <= 0:
        return f"[{'-' * width}]   0.0% | evaluation"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - started_at, 0.001)
    avg = elapsed / max(done, 1)
    remaining = max(total - done, 0)
    eta = int(avg * remaining)
    return f"[{bar}] {ratio*100:5.1f}% | {done}/{total} | ETA {eta:>4}s | evaluation"


def run_evaluation(
    input_file: str,
    output_file: str,
    judge_model: str = "openai/gpt-4o-mini",
    judge_prompt_version: str = "trap_v2",
    judge_prompt_template: str | None = None,
    judge_throttle_seconds: float = 0.0,
    judge_max_retries: int = 2,
    judge_parallel_workers: int = 1,
    runs_root: str = "./eval_data/runs",
    run_uuid: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = read_excel(input_file)
    artifacts = init_run_artifacts(runs_root, run_uuid=run_uuid)
    total_rows = len(df)
    started_at = time.time()
    print(f"Evaluation started: input={input_file} rows={total_rows} judge_model={judge_model}")
    prompt_hash = _judge_prompt_hash(judge_prompt_template)
    if judge_prompt_template:
        print(f"Judge prompt: CUSTOM (version={judge_prompt_version})")
    else:
        print(f"Judge prompt: DEFAULT (version={judge_prompt_version})")
    print(f"Judge prompt hash: {prompt_hash[:12]}")
    if int(judge_parallel_workers or 1) > 1:
        print(
            "Hinweis: judge_parallel_workers ist konfiguriert, "
            "die Judge-Evaluation laeuft in dieser Version aber weiterhin seriell."
        )
    raw_llm = get_llm(model_name=judge_model, temperature=0)
    pricing_map, pricing_details = resolve_pricing_for_models(
        models=[judge_model],
        default_pricing={},
        cache_path="./eval_data/pricing_cache.json",
    )

    scores = []
    reasons = []
    coverages = []
    judge_input_tokens = []
    judge_output_tokens = []
    judge_cost_usd = []
    supported_counts = []
    contradicted_counts = []
    not_in_scope_counts = []
    total_claim_counts = []
    support_rates = []
    contradiction_rates = []
    not_in_scope_rates = []
    atomic_coverage_scores = []
    predicted_supported_counts = []
    predicted_unsupported_counts = []
    predicted_total_counts = []
    atomic_precisions = []
    hallucination_rates = []
    trap_flags = []
    trap_pass_values = []
    trap_hallucination_values = []
    trap_fallibility_values = []
    confidence_means = []
    confidence_found_flags = []
    evaluation_cache, evaluation_key_cols = _load_evaluation_cache(
        output_file,
        df,
        judge_model,
        judge_prompt_version,
        prompt_hash,
    )
    reused_judge_rows = 0
    skipped_benchmark_error_rows = 0
    actual_judge_calls = 0

    print(f"Evaluation progress: 0/{total_rows} rows")
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        print("\r" + _render_eval_progress(idx - 1, total_rows, started_at), end="", flush=True)
        generated = row.get("KI_Antwort", "")
        question = row.get("Frage", "")
        answer_type = row.get("Antworttyp", "")
        reference_answer = row.get("Referenzantwort", "")
        expected_trap_behavior = row.get("Expected_Trap_Behavior", row.get("expected_trap_behavior", ""))
        atomic_facts = parse_atomic_facts_from_row(row)
        confidence_mean, confidence_found_flag = extract_confidence_mean(generated)
        trap_mode = is_trap_question(str(answer_type), str(reference_answer))
        if trap_mode and not str(expected_trap_behavior or "").strip():
            expected_trap_behavior = f"Goldstandard-Einschraenkung: {reference_answer}"
        benchmark_error = _cell_text(row.get("error", ""))
        score = math.nan
        reason = ""
        coverage = math.nan
        input_tokens = 0
        output_tokens = 0
        call_cost = 0.0
        supported = 0
        contradicted = 0
        not_in_scope = 0
        predicted_supported = 0
        predicted_unsupported = 0
        predicted_total = 0
        trap_pass = math.nan

        cached = None
        if evaluation_cache and evaluation_key_cols:
            cached = evaluation_cache.get(_row_key(row, evaluation_key_cols))
            if cached is not None and not _source_fields_match(row, cached):
                cached = None
        if cached is not None:
            reused_judge_rows += 1
            score = _to_float(cached.get("Likert_Score", math.nan))
            reason = str(cached.get("Judge_Begruendung", ""))
            coverage = _to_float(cached.get("Atomic_Facts_Coverage", math.nan))
            input_tokens = _to_int(cached.get("judge_input_tokens", 0))
            output_tokens = _to_int(cached.get("judge_output_tokens", 0))
            call_cost = _to_float(cached.get("judge_cost_usd", 0.0), 0.0)
            supported = _to_int(cached.get("Supported_Claims", 0))
            contradicted = _to_int(cached.get("Contradicted_Claims", 0))
            not_in_scope = _to_int(cached.get("Not_In_Scope_Claims", 0))
            predicted_supported = _to_int(cached.get("Predicted_Supported_Claims", 0))
            predicted_unsupported = _to_int(cached.get("Predicted_Unsupported_Claims", 0))
            predicted_total = _to_int(cached.get("Predicted_Total_Claims", 0))
            trap_pass = _to_float(cached.get("Trap_Pass", math.nan))
            trap_mode = bool(_to_int(cached.get("Is_Trap_Question", int(trap_mode))))
        elif benchmark_error:
            skipped_benchmark_error_rows += 1
            reason = f"Benchmark API Error: {benchmark_error}"
        else:
            prompt = build_atomic_facts_judge_prompt(
                question=question,
                generated=generated,
                atomic_facts=atomic_facts,
                judge_prompt_template=judge_prompt_template,
                trap_mode=trap_mode,
                expected_trap_behavior=str(expected_trap_behavior or ""),
            )
            try:  # pragma: no cover - network dependent
                if trap_mode:
                    output_schema = (
                        '{"score": <0-4>, "begruendung": "<kurz>", "coverage": <0..1>, '
                        '"supported": <int>, "contradicted": <int>, "not_in_scope": <int>, '
                        '"predicted_supported_claims": <int>, "predicted_unsupported_claims": <int>, '
                        '"predicted_total_claims": <int>, '
                        '"trap_pass": <0|1>}'
                    )
                else:
                    output_schema = (
                        '{"score": <0-4>, "begruendung": "<kurz>", "coverage": <0..1>, '
                        '"supported": <int>, "contradicted": <int>, "not_in_scope": <int>, '
                        '"predicted_supported_claims": <int>, "predicted_unsupported_claims": <int>, '
                        '"predicted_total_claims": <int>}'
                    )
                full_prompt = (
                    prompt
                    + "\n\nGib NUR valides JSON mit genau diesen Feldern zurueck: "
                    + output_schema
                )
                if trap_mode:
                    full_prompt += "\nBei Fangfragen muss `trap_pass` immer explizit 0 oder 1 sein."
                if judge_throttle_seconds > 0 and actual_judge_calls > 0:
                    time.sleep(judge_throttle_seconds)
                actual_judge_calls += 1
                result = invoke_with_retry(
                    raw_llm,
                    full_prompt,
                    max_retries=judge_max_retries,
                    throttle_seconds=judge_throttle_seconds,
                )
                if result.get("error"):
                    raise RuntimeError(result["error"])
                input_tokens = int(result.get("input_tokens", 0) or 0)
                output_tokens = int(result.get("output_tokens", 0) or 0)
                call_cost = calculate_cost(judge_model, input_tokens, output_tokens, pricing_map)
                raw_response = result.get("content", "")
                score, reason, coverage = parse_judge_json(raw_response)
                supported, contradicted, not_in_scope, trap_pass = parse_claim_labels_json(raw_response)
                predicted_supported, predicted_unsupported, predicted_total = parse_predicted_claim_labels_json(
                    raw_response
                )
            except Exception as exc:
                reason = f"API Error: {exc}"
                supported = 0
                contradicted = 0
                not_in_scope = 0
                predicted_supported = 0
                predicted_unsupported = 0
                predicted_total = 0
                trap_pass = math.nan

        total_claims = supported + contradicted + not_in_scope
        safe_total = max(total_claims, 1)
        support_rate = supported / safe_total
        contradiction_rate = contradicted / safe_total
        not_in_scope_rate = not_in_scope / safe_total
        atomic_coverage_score = support_rate
        if predicted_total <= 0 and (predicted_supported + predicted_unsupported) > 0:
            predicted_total = predicted_supported + predicted_unsupported
        if predicted_total > 0:
            atomic_precision = predicted_supported / predicted_total
            hallucination_rate = 1.0 - atomic_precision
        else:
            atomic_precision = math.nan
            hallucination_rate = math.nan
        derived_score = derive_likert_from_claim_counts(supported, contradicted, not_in_scope)
        if isinstance(score, float) and math.isnan(score) and not math.isnan(derived_score):
            score = derived_score
        if isinstance(coverage, float) and math.isnan(coverage) and total_claims > 0:
            coverage = support_rate
        if isinstance(score, float) and math.isnan(score):
            reason = reason or "Judge failed"
        trap_hallucination = (
            1.0 - trap_pass if trap_mode and not (isinstance(trap_pass, float) and math.isnan(trap_pass)) else math.nan
        )
        # Primary weakness metric: probability of falling for a trap question.
        trap_fallibility = trap_hallucination

        scores.append(score)
        reasons.append(sanitize_judge_text(reason))
        coverages.append(coverage)
        judge_input_tokens.append(input_tokens)
        judge_output_tokens.append(output_tokens)
        judge_cost_usd.append(call_cost)
        supported_counts.append(supported)
        contradicted_counts.append(contradicted)
        not_in_scope_counts.append(not_in_scope)
        total_claim_counts.append(total_claims)
        support_rates.append(support_rate)
        contradiction_rates.append(contradiction_rate)
        not_in_scope_rates.append(not_in_scope_rate)
        atomic_coverage_scores.append(atomic_coverage_score)
        predicted_supported_counts.append(predicted_supported)
        predicted_unsupported_counts.append(predicted_unsupported)
        predicted_total_counts.append(predicted_total)
        atomic_precisions.append(atomic_precision)
        hallucination_rates.append(hallucination_rate)
        trap_flags.append(int(trap_mode))
        trap_pass_values.append(trap_pass if trap_mode else math.nan)
        trap_hallucination_values.append(trap_hallucination)
        trap_fallibility_values.append(trap_fallibility)
        confidence_means.append(confidence_mean)
        confidence_found_flags.append(int(confidence_found_flag))
    if total_rows:
        print("\r" + _render_eval_progress(total_rows, total_rows, started_at), flush=True)
    else:
        print(_render_eval_progress(0, 0, started_at), flush=True)

    df["Likert_Score"] = scores
    df["Judge_Begruendung"] = reasons
    df["Atomic_Facts_Coverage"] = coverages
    df["judge_model"] = judge_model
    df["judge_prompt_version"] = judge_prompt_version
    df["judge_prompt_hash"] = prompt_hash
    df["judge_input_tokens"] = judge_input_tokens
    df["judge_output_tokens"] = judge_output_tokens
    df["judge_cost_usd"] = judge_cost_usd
    df["Supported_Claims"] = supported_counts
    df["Contradicted_Claims"] = contradicted_counts
    df["Not_In_Scope_Claims"] = not_in_scope_counts
    df["Total_Claims"] = total_claim_counts
    df["Atomic_Coverage_Score"] = atomic_coverage_scores
    df["Support_Rate"] = support_rates
    df["Contradiction_Rate"] = contradiction_rates
    df["Not_In_Scope_Rate"] = not_in_scope_rates
    df["Predicted_Supported_Claims"] = predicted_supported_counts
    df["Predicted_Unsupported_Claims"] = predicted_unsupported_counts
    df["Predicted_Total_Claims"] = predicted_total_counts
    df["Atomic_Precision"] = atomic_precisions
    df["Hallucination_Rate"] = hallucination_rates
    df["Is_Trap_Question"] = trap_flags
    df["Trap_Pass"] = trap_pass_values
    df["Trap_Hallucination"] = trap_hallucination_values
    df["Trap_Fallibility"] = trap_fallibility_values
    df["Confidence_Mean"] = confidence_means
    df["Confidence_Found_Flag"] = confidence_found_flags
    write_excel(output_file, df)

    eval_rows = []
    for idx, (score, reason) in enumerate(zip(scores, reasons)):
        base_row = EvaluationRow(
            likert_score=float(score)
            if not (isinstance(score, float) and math.isnan(score))
            else float("nan"),
            judge_reason=reason,
            judge_model=judge_model,
            judge_prompt_version=judge_prompt_version,
        ).__dict__
        base_row["judge_prompt_hash"] = prompt_hash
        base_row["judge_input_tokens"] = int(judge_input_tokens[idx])
        base_row["judge_output_tokens"] = int(judge_output_tokens[idx])
        base_row["judge_cost_usd"] = float(judge_cost_usd[idx])
        base_row["supported_claims"] = int(supported_counts[idx])
        base_row["contradicted_claims"] = int(contradicted_counts[idx])
        base_row["not_in_scope_claims"] = int(not_in_scope_counts[idx])
        base_row["total_claims"] = int(total_claim_counts[idx])
        base_row["atomic_coverage_score"] = float(atomic_coverage_scores[idx])
        base_row["support_rate"] = float(support_rates[idx])
        base_row["contradiction_rate"] = float(contradiction_rates[idx])
        base_row["not_in_scope_rate"] = float(not_in_scope_rates[idx])
        base_row["predicted_supported_claims"] = int(predicted_supported_counts[idx])
        base_row["predicted_unsupported_claims"] = int(predicted_unsupported_counts[idx])
        base_row["predicted_total_claims"] = int(predicted_total_counts[idx])
        base_row["atomic_precision"] = (
            float(atomic_precisions[idx]) if not math.isnan(atomic_precisions[idx]) else float("nan")
        )
        base_row["hallucination_rate"] = (
            float(hallucination_rates[idx]) if not math.isnan(hallucination_rates[idx]) else float("nan")
        )
        base_row["is_trap_question"] = int(trap_flags[idx])
        base_row["trap_pass"] = float(trap_pass_values[idx]) if not math.isnan(trap_pass_values[idx]) else float("nan")
        base_row["trap_hallucination"] = (
            float(trap_hallucination_values[idx])
            if not math.isnan(trap_hallucination_values[idx])
            else float("nan")
        )
        base_row["trap_fallibility"] = (
            float(trap_fallibility_values[idx]) if not math.isnan(trap_fallibility_values[idx]) else float("nan")
        )
        base_row["confidence_mean"] = (
            float(confidence_means[idx]) if not math.isnan(confidence_means[idx]) else float("nan")
        )
        base_row["confidence_found_flag"] = int(confidence_found_flags[idx])
        eval_rows.append(base_row)
    write_jsonl(artifacts["evaluated_results_jsonl"], eval_rows)

    stage_cfg_path = os.path.join(artifacts["run_dir"], "evaluation.config.json")
    judge_totals = {
        "rows": len(df),
        "judge_input_tokens_total": int(sum(judge_input_tokens)),
        "judge_output_tokens_total": int(sum(judge_output_tokens)),
        "judge_cost_usd_total": round(float(sum(judge_cost_usd)), 6),
        "judge_input_tokens_avg": round((sum(judge_input_tokens) / max(len(judge_input_tokens), 1)), 3),
        "judge_output_tokens_avg": round((sum(judge_output_tokens) / max(len(judge_output_tokens), 1)), 3),
        "judge_cost_usd_avg": round((sum(judge_cost_usd) / max(len(judge_cost_usd), 1)), 6),
        "supported_claims_total": int(sum(supported_counts)),
        "contradicted_claims_total": int(sum(contradicted_counts)),
        "not_in_scope_claims_total": int(sum(not_in_scope_counts)),
        "total_claims_total": int(sum(total_claim_counts)),
        "predicted_supported_claims_total": int(sum(predicted_supported_counts)),
        "predicted_unsupported_claims_total": int(sum(predicted_unsupported_counts)),
        "predicted_total_claims_total": int(sum(predicted_total_counts)),
        "atomic_coverage_score_avg": round(float(sum(atomic_coverage_scores) / max(len(atomic_coverage_scores), 1)), 6),
        "support_rate_avg": round(float(sum(support_rates) / max(len(support_rates), 1)), 6),
        "contradiction_rate_avg": round(float(sum(contradiction_rates) / max(len(contradiction_rates), 1)), 6),
        "not_in_scope_rate_avg": round(float(sum(not_in_scope_rates) / max(len(not_in_scope_rates), 1)), 6),
        "atomic_precision_avg": round(
            float(sum(v for v in atomic_precisions if not math.isnan(v)) / max(sum(not math.isnan(v) for v in atomic_precisions), 1)),
            6,
        )
        if any(not math.isnan(v) for v in atomic_precisions)
        else float("nan"),
        "hallucination_rate_avg": round(
            float(sum(v for v in hallucination_rates if not math.isnan(v)) / max(sum(not math.isnan(v) for v in hallucination_rates), 1)),
            6,
        )
        if any(not math.isnan(v) for v in hallucination_rates)
        else float("nan"),
        "trap_questions": int(sum(trap_flags)),
    }
    print(
        "Judge usage totals: "
        f"in={judge_totals['judge_input_tokens_total']} "
        f"out={judge_totals['judge_output_tokens_total']} "
        f"cost_usd={judge_totals['judge_cost_usd_total']}"
    )
    if reused_judge_rows or skipped_benchmark_error_rows:
        print(
            "Evaluation resume: "
            f"reused_judge_rows={reused_judge_rows} "
            f"skipped_benchmark_error_rows={skipped_benchmark_error_rows} "
            f"actual_judge_calls={actual_judge_calls}"
        )
    print(f"Evaluation completed: output={output_file} rows={len(df)}")
    write_json(
        stage_cfg_path,
        {
            "stage": "evaluation",
            "input_file": input_file,
            "output_file": output_file,
            "judge_model": judge_model,
            "judge_prompt_version": judge_prompt_version,
            "judge_prompt_hash": prompt_hash,
            "judge_throttle_seconds": judge_throttle_seconds,
            "judge_max_retries": judge_max_retries,
            "judge_prompt_template_present": bool(judge_prompt_template),
            "judge_pricing_details": pricing_details,
            "judge_usage_totals": judge_totals,
            "reused_judge_rows": reused_judge_rows,
            "skipped_benchmark_error_rows": skipped_benchmark_error_rows,
        },
    )
    return df, artifacts
