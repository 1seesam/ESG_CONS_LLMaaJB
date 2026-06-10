from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Ensure the repository root is on sys.path when this script is executed from tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarking.analysis_pipeline import run_analysis
from src.benchmarking.io import dataframe_from_rows, read_jsonl, write_excel, write_json, write_jsonl
from src.benchmarking.visualization import generate_run_figures


KEY_COLUMNS = ["Run_ID", "Model", "Method", "ID"]
RAW_KEY_COLUMNS = ["run_number", "model", "method", "question_id"]


def _parse_ids(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _key(row: pd.Series, columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in columns)


def _raw_key(row: dict, columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in columns)


def _find_evaluated_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "results_evaluated_scientific_FINAL.xlsx",
        run_dir / "results_evaluated_scientific.xlsx",
        run_dir / "results_evaluated_scientific_merged.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No evaluated Excel file found in "
        f"{run_dir}. Pass --base-evaluated/--patch-evaluated explicitly."
    )


def _read_evaluated(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    for sheet_name in ("RUN_RESULTS", "Sheet1"):
        if sheet_name in sheets and set(KEY_COLUMNS).issubset(sheets[sheet_name].columns):
            return sheets[sheet_name].copy()
    first = next(iter(sheets.values()))
    if not set(KEY_COLUMNS).issubset(first.columns):
        raise ValueError(f"{path} does not contain key columns {KEY_COLUMNS}")
    return first.copy()


def _write_compat_results_excel(path: Path, raw_rows: list[dict]) -> None:
    df = dataframe_from_rows(raw_rows)
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
    write_excel(str(path), compat_df)


def _merge_questions(base_run_dir: Path, patch_question_file: Path, out_path: Path, replace_ids: set[str]) -> None:
    base_questions_path = base_run_dir / "questions.json"
    if not base_questions_path.exists():
        return
    base_questions = json.loads(base_questions_path.read_text(encoding="utf-8"))
    patch_questions = json.loads(patch_question_file.read_text(encoding="utf-8")) if patch_question_file.exists() else []
    patch_by_id = {str(item.get("id", "")): item for item in patch_questions}
    merged = []
    for item in base_questions:
        qid = str(item.get("id", ""))
        merged.append(patch_by_id.get(qid, item) if qid in replace_ids else item)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a focused patch run into a base benchmark run and re-run analysis.")
    parser.add_argument("--base-run", required=True, help="Run ID used as the main result base.")
    parser.add_argument("--patch-run", required=True, help="Run ID containing corrected replacement rows.")
    parser.add_argument("--output-run", required=True, help="Run ID for the merged output.")
    parser.add_argument("--replace-ids", required=True, help="Comma-separated question IDs to replace from the patch run.")
    parser.add_argument("--runs-root", default="./eval_data/runs")
    parser.add_argument("--base-evaluated", default=None)
    parser.add_argument("--patch-evaluated", default=None)
    parser.add_argument("--patch-question-file", default="./eval_data/questions_patch_ragfix_2026-05-19.json")
    parser.add_argument("--baseline-method", default="Zero-Shot")
    parser.add_argument("--time-cost-factor", type=float, default=2.0)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    base_dir = runs_root / args.base_run
    patch_dir = runs_root / args.patch_run
    out_dir = runs_root / args.output_run
    out_dir.mkdir(parents=True, exist_ok=True)

    replace_ids = _parse_ids(args.replace_ids)
    base_eval_path = Path(args.base_evaluated) if args.base_evaluated else _find_evaluated_file(base_dir)
    patch_eval_path = Path(args.patch_evaluated) if args.patch_evaluated else _find_evaluated_file(patch_dir)

    base_df = _read_evaluated(base_eval_path)
    patch_df = _read_evaluated(patch_eval_path)
    patch_df = patch_df[patch_df["ID"].astype(str).isin(replace_ids)].copy()
    patch_by_key = {_key(row, KEY_COLUMNS): row for _, row in patch_df.iterrows()}

    merged_rows = []
    replaced_keys: set[tuple[str, ...]] = set()
    for _, row in base_df.iterrows():
        row_key = _key(row, KEY_COLUMNS)
        if str(row.get("ID", "")).strip() in replace_ids and row_key in patch_by_key:
            merged_rows.append(patch_by_key[row_key])
            replaced_keys.add(row_key)
        else:
            merged_rows.append(row)
    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)

    evaluated_out = out_dir / "results_evaluated_scientific_merged.xlsx"
    write_excel(str(evaluated_out), merged_df)

    raw_replaced = 0
    raw_out = out_dir / "raw_results.jsonl"
    base_raw = base_dir / "raw_results.jsonl"
    patch_raw = patch_dir / "raw_results.jsonl"
    if base_raw.exists() and patch_raw.exists():
        base_raw_rows = read_jsonl(str(base_raw))
        patch_raw_rows = [row for row in read_jsonl(str(patch_raw)) if str(row.get("question_id", "")) in replace_ids]
        patch_raw_by_key = {_raw_key(row, RAW_KEY_COLUMNS): row for row in patch_raw_rows}
        merged_raw = []
        for row in base_raw_rows:
            row_key = _raw_key(row, RAW_KEY_COLUMNS)
            if str(row.get("question_id", "")) in replace_ids and row_key in patch_raw_by_key:
                merged_raw.append(patch_raw_by_key[row_key])
                raw_replaced += 1
            else:
                merged_raw.append(row)
        write_jsonl(str(raw_out), merged_raw)
        _write_compat_results_excel(out_dir / "results_benchmark_final_merged.xlsx", merged_raw)

    question_file = out_dir / "questions.json"
    _merge_questions(base_dir, Path(args.patch_question_file), question_file, replace_ids)
    if not question_file.exists() and (base_dir / "questions.json").exists():
        shutil.copy2(base_dir / "questions.json", question_file)

    stats_df, analysis_artifacts = run_analysis(
        input_file=str(evaluated_out),
        baseline_method=args.baseline_method,
        runs_root=str(runs_root),
        time_cost_factor=args.time_cost_factor,
        run_uuid=args.output_run,
    )
    metrics_csv = out_dir / "metrics_summary.csv"
    stats_df.to_csv(metrics_csv, index=False)

    figure_files: list[str] = []
    figures_dir = out_dir / "thesis_figures"
    if not args.no_figures and question_file.exists():
        figure_files = generate_run_figures(merged_df, question_file, figures_dir)

    missing_patch_ids = sorted(
        replace_ids - {str(value) for value in patch_df["ID"].dropna().astype(str).unique()},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    summary = {
        "run_uuid": args.output_run,
        "merge_base_run": args.base_run,
        "merge_patch_run": args.patch_run,
        "replace_ids": sorted(replace_ids, key=lambda x: int(x) if x.isdigit() else x),
        "base_evaluated": str(base_eval_path),
        "patch_evaluated": str(patch_eval_path),
        "evaluation_output": str(evaluated_out),
        "benchmark_output": str(out_dir / "results_benchmark_final_merged.xlsx") if raw_out.exists() else None,
        "raw_results_jsonl": str(raw_out) if raw_out.exists() else None,
        "metrics_summary_csv": str(metrics_csv),
        "stats_report_json": analysis_artifacts["stats_report_json"],
        "summary_main_tests_csv": analysis_artifacts["summary_main_tests_csv"],
        "summary_trap_questions_csv": analysis_artifacts["summary_trap_questions_csv"],
        "figures_dir": str(figures_dir) if figure_files else None,
        "figure_files": figure_files,
        "merged_rows": int(len(merged_df)),
        "replaced_evaluated_rows": int(len(replaced_keys)),
        "replaced_raw_rows": int(raw_replaced),
        "missing_patch_ids": missing_patch_ids,
    }
    write_json(str(out_dir / "pipeline.summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
