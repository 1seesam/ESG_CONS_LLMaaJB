from __future__ import annotations

import os
from typing import Any, Dict

from .analysis_pipeline import run_analysis
from .config_schema import PathsConfig, RunConfig
from .dataset import load_questions
from .evaluation_pipeline import run_evaluation
from .io import read_excel, write_json
from .logging_utils import default_run_uuid
from .pipeline import run_benchmark
from .visualization import generate_run_figures


JUDGE_ISSUE_MARKERS = (
    "API Error",
    "Judge JSON parse failed",
    "Judge returned empty response",
)


def _safe_text(value: object, limit: int = 260) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] + "..." if len(text) > limit else text


def _sample_rows(df, mask, columns: list[str], max_items: int = 8) -> list[dict[str, str]]:
    if df.empty or not mask.any():
        return []
    available = [column for column in columns if column in df.columns]
    if not available:
        return []
    rows: list[dict[str, str]] = []
    for _, row in df.loc[mask, available].head(max_items).iterrows():
        rows.append({column: _safe_text(row.get(column, "")) for column in available})
    return rows


def _expected_benchmark_rows(run_config: RunConfig) -> int | None:
    try:
        questions = load_questions(run_config.question_file)
    except Exception:
        return None
    return len(questions) * len(run_config.models) * len(run_config.methods) * int(run_config.num_runs)


def _build_data_quality_report(run_config: RunConfig, benchmark_df, evaluated_df) -> dict[str, Any]:
    expected_rows = _expected_benchmark_rows(run_config)
    benchmark_error_mask = (
        benchmark_df["error"].fillna("").astype(str).str.strip().ne("")
        if "error" in benchmark_df.columns
        else benchmark_df.index.to_series().eq("__never__")
    )
    benchmark_error_rows = int(benchmark_error_mask.sum())
    missing_benchmark_rows = (
        max(int(expected_rows) - int(len(benchmark_df)), 0) if expected_rows is not None else 0
    )

    if "Judge_Begruendung" in evaluated_df.columns:
        reason_text = evaluated_df["Judge_Begruendung"].fillna("").astype(str)
        judge_issue_mask = reason_text.apply(
            lambda text: any(marker.lower() in text.lower() for marker in JUDGE_ISSUE_MARKERS)
        )
    else:
        judge_issue_mask = evaluated_df.index.to_series().eq("__never__")
    judge_issue_rows = int(judge_issue_mask.sum())

    missing_score_mask = (
        evaluated_df["Likert_Score"].isna()
        if "Likert_Score" in evaluated_df.columns
        else evaluated_df.index.to_series().eq("__never__")
    )
    missing_score_rows = int(missing_score_mask.sum())

    stats_affected = bool(benchmark_error_rows or missing_benchmark_rows or judge_issue_rows or missing_score_rows)
    return {
        "status": "warning" if stats_affected else "ok",
        "stats_may_be_affected": stats_affected,
        "expected_benchmark_rows": expected_rows,
        "benchmark_rows": int(len(benchmark_df)),
        "missing_benchmark_rows": missing_benchmark_rows,
        "benchmark_api_error_rows": benchmark_error_rows,
        "judge_issue_rows": judge_issue_rows,
        "missing_likert_score_rows": missing_score_rows,
        "benchmark_error_examples": _sample_rows(
            benchmark_df,
            benchmark_error_mask,
            ["model", "method", "question_id", "run_number", "answer_type", "error"],
        ),
        "judge_issue_examples": _sample_rows(
            evaluated_df,
            judge_issue_mask | missing_score_mask,
            ["Model", "Method", "ID", "Run_ID", "Antworttyp", "Judge_Begruendung"],
        ),
    }


def run_end_to_end(
    run_config: RunConfig,
    evaluated_output_file: str = "./eval_data/results_evaluated_scientific.xlsx",
    judge_model: str = "openai/gpt-4o-mini",
    judge_prompt_version: str = "trap_v2",
    judge_prompt_template: str | None = None,
    judge_throttle_seconds: float = 0.0,
    judge_max_retries: int = 2,
    judge_parallel_workers: int = 1,
    baseline_method: str = "Zero-Shot",
    time_cost_factor: float = 0.002,
    runs_root: str = "./eval_data/runs",
    run_uuid: str | None = None,
    enable_overconfidence: bool = False,
    enable_visualizations: bool = False,
    enable_robustness: bool = False,
) -> Dict[str, object]:
    run_uuid = run_uuid or default_run_uuid()

    benchmark_df, benchmark_artifacts = run_benchmark(
        run_config,
        paths=PathsConfig(runs_root=runs_root),
        run_uuid=run_uuid,
    )
    if benchmark_df.empty:
        raise RuntimeError("Benchmark produced no rows; stopping pipeline.")

    eval_df, eval_artifacts = run_evaluation(
        input_file=run_config.results_file,
        output_file=evaluated_output_file,
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
        judge_prompt_template=judge_prompt_template,
        judge_throttle_seconds=judge_throttle_seconds,
        judge_max_retries=judge_max_retries,
        judge_parallel_workers=judge_parallel_workers,
        runs_root=runs_root,
        run_uuid=run_uuid,
    )
    stats_df, analysis_artifacts = run_analysis(
        input_file=evaluated_output_file,
        baseline_method=baseline_method,
        runs_root=runs_root,
        time_cost_factor=time_cost_factor,
        run_uuid=run_uuid,
        enable_overconfidence=enable_overconfidence,
        enable_robustness=enable_robustness,
    )

    run_dir = benchmark_artifacts["run_dir"]
    metrics_csv = os.path.join(run_dir, "metrics_summary.csv")
    stats_df.to_csv(metrics_csv, index=False)
    figures_dir = os.path.join(run_dir, "thesis_figures")
    generated_figures: list[str] = []
    if enable_visualizations:
        figure_eval_df = read_excel(evaluated_output_file)
        generated_figures = generate_run_figures(figure_eval_df, run_config.question_file, figures_dir)
    tasktype_summary_csv = os.path.join(figures_dir, "summary_tasktype_deltas.csv") if enable_visualizations else None
    precision_tasktype_summary_csv = (
        os.path.join(figures_dir, "summary_tasktype_deltas_atomic_precision.csv") if enable_visualizations else None
    )
    hallucination_tasktype_summary_csv = (
        os.path.join(figures_dir, "summary_tasktype_deltas_hallucination_rate.csv") if enable_visualizations else None
    )
    run_variance_summary_csv = (
        os.path.join(figures_dir, "summary_run_variance_metrics.csv") if enable_visualizations else None
    )
    data_quality = _build_data_quality_report(run_config, benchmark_df, eval_df)

    summary_path = os.path.join(run_dir, "pipeline.summary.json")
    summary = {
        "run_uuid": run_uuid,
        "run_dir": run_dir,
        "benchmark_rows": len(benchmark_df),
        "benchmark_output": run_config.results_file,
        "evaluation_output": evaluated_output_file,
        "stats_report_json": analysis_artifacts["stats_report_json"],
        "metrics_summary_csv": metrics_csv,
        "summary_main_tests_csv": analysis_artifacts.get("summary_main_tests_csv"),
        "summary_trap_questions_csv": analysis_artifacts.get("summary_trap_questions_csv"),
        "summary_robustness_csv": analysis_artifacts.get("summary_robustness_csv") if enable_robustness else None,
        "summary_overconfidence_csv": (
            analysis_artifacts.get("summary_overconfidence_csv") if enable_overconfidence else None
        ),
        "summary_tasktype_deltas_csv": tasktype_summary_csv if enable_visualizations and os.path.exists(tasktype_summary_csv) else None,
        "summary_tasktype_deltas_atomic_precision_csv": (
            precision_tasktype_summary_csv
            if enable_visualizations and os.path.exists(precision_tasktype_summary_csv)
            else None
        ),
        "summary_tasktype_deltas_hallucination_rate_csv": (
            hallucination_tasktype_summary_csv
            if enable_visualizations and os.path.exists(hallucination_tasktype_summary_csv)
            else None
        ),
        "summary_run_variance_metrics_csv": (
            run_variance_summary_csv if enable_visualizations and os.path.exists(run_variance_summary_csv) else None
        ),
        "figures_dir": figures_dir if enable_visualizations else None,
        "figure_files": generated_figures,
        "raw_results_jsonl": benchmark_artifacts["raw_results_jsonl"],
        "evaluated_results_jsonl": eval_artifacts["evaluated_results_jsonl"],
        "pipeline_summary_json": summary_path,
        "data_quality": data_quality,
    }
    write_json(summary_path, summary)
    return summary
