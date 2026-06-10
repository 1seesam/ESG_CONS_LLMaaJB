import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _reexec_with_project_venv_if_needed() -> None:
    venv_python = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else PROJECT_ROOT / ".venv" / "bin" / "python"
    )
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python == target_python:
        return

    if os.environ.get("THESIS_PIPELINE_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["THESIS_PIPELINE_REEXEC"] = "1"
    argv = [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    completed = subprocess.run(argv, env=env, cwd=str(PROJECT_ROOT))
    raise SystemExit(completed.returncode)


def _ensure_project_cwd() -> None:
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass


if __name__ == "__main__":
    _ensure_project_cwd()
    _reexec_with_project_venv_if_needed()

from src.benchmarking.config_schema import RunConfig
from src.benchmarking.master_config import (
    create_master_config_template,
    load_master_config,
)
from src.benchmarking.logging_utils import default_run_uuid
from src.benchmarking.orchestrator import run_end_to_end
from src.convert_questions import excel_to_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-click end-to-end pipeline: benchmark -> evaluation -> analysis."
    )
    parser.add_argument("--master-config", default="./eval_data/master_config.xlsx")
    parser.add_argument(
        "--init-master-config",
        action="store_true",
        help="Create a template master config Excel and exit.",
    )

    parser.add_argument("--input-file", default=None)
    parser.add_argument("--catalog-excel", default=None)
    parser.add_argument("--auto-convert-questions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--benchmark-output", default=None)
    parser.add_argument("--evaluated-output", default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--throttle-seconds", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--use-rag", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-prompt-version", default=None)
    parser.add_argument("--judge-prompt-path", default=None)
    parser.add_argument("--judge-throttle-seconds", type=float, default=None)
    parser.add_argument("--judge-max-retries", type=int, default=None)
    parser.add_argument("--prompt-zero-shot-path", default=None)
    parser.add_argument("--prompt-advanced-path", default=None)
    parser.add_argument("--prompt-rag-path", default=None)
    parser.add_argument("--enable-overconfidence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-visualizations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-robustness", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--baseline-method", default=None)
    parser.add_argument("--time-cost-factor", type=float, default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--run-uuid", default=None)
    parser.add_argument("--default-max-words", type=int, default=None)
    parser.add_argument("--auto-reindex", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rag-chunk-size", type=int, default=None)
    parser.add_argument("--rag-chunk-overlap", type=int, default=None)
    parser.add_argument("--max-rag-context-chars", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--slow-call-seconds", type=float, default=None)
    parser.add_argument("--max-consecutive-api-fails", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--judge-parallel-workers", type=int, default=None)
    parser.add_argument("--auto-resume-until-complete", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-auto-resume-attempts", type=int, default=None)
    parser.add_argument("--auto-resume-sleep-seconds", type=float, default=None)
    return parser.parse_args()


def _resolve_config(args):
    cfg = load_master_config(args.master_config)

    overrides = {
        "input_file": args.input_file,
        "catalog_excel": args.catalog_excel,
        "auto_convert_questions": args.auto_convert_questions,
        "benchmark_output": args.benchmark_output,
        "evaluated_output": args.evaluated_output,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "temperature": args.temperature,
        "methods": args.methods,
        "models": args.models,
        "throttle_seconds": args.throttle_seconds,
        "max_retries": args.max_retries,
        "use_rag": args.use_rag,
        "judge_model": args.judge_model,
        "judge_prompt_version": args.judge_prompt_version,
        "judge_prompt_path": args.judge_prompt_path,
        "judge_throttle_seconds": args.judge_throttle_seconds,
        "judge_max_retries": args.judge_max_retries,
        "prompt_zero_shot_path": args.prompt_zero_shot_path,
        "prompt_advanced_path": args.prompt_advanced_path,
        "prompt_rag_path": args.prompt_rag_path,
        "enable_overconfidence": args.enable_overconfidence,
        "enable_visualizations": args.enable_visualizations,
        "enable_robustness": args.enable_robustness,
        "baseline_method": args.baseline_method,
        "time_cost_factor": args.time_cost_factor,
        "runs_root": args.runs_root,
        "run_uuid": args.run_uuid,
        "default_max_words": args.default_max_words,
        "auto_reindex": args.auto_reindex,
        "rag_chunk_size": args.rag_chunk_size,
        "rag_chunk_overlap": args.rag_chunk_overlap,
        "max_rag_context_chars": args.max_rag_context_chars,
        "max_prompt_chars": args.max_prompt_chars,
        "slow_call_seconds": args.slow_call_seconds,
        "max_consecutive_api_fails": args.max_consecutive_api_fails,
        "parallel_workers": args.parallel_workers,
        "judge_parallel_workers": args.judge_parallel_workers,
        "auto_resume_until_complete": args.auto_resume_until_complete,
        "max_auto_resume_attempts": args.max_auto_resume_attempts,
        "auto_resume_sleep_seconds": args.auto_resume_sleep_seconds,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def _print_data_quality_summary(summary):
    quality = summary.get("data_quality") or {}
    if not isinstance(quality, dict):
        return

    print("data_quality_status:", quality.get("status", "unknown"))
    if not quality.get("stats_may_be_affected"):
        print("data_quality: OK - keine Benchmark-/Judge-Fehler erkannt.")
        return

    print("WARNING: API/Judge-Probleme koennen Statistik und Grafiken beeinflussen.")
    print(f"expected_benchmark_rows: {quality.get('expected_benchmark_rows')}")
    print(f"benchmark_rows: {quality.get('benchmark_rows')}")
    print(f"missing_benchmark_rows: {quality.get('missing_benchmark_rows')}")
    print(f"benchmark_api_error_rows: {quality.get('benchmark_api_error_rows')}")
    print(f"judge_issue_rows: {quality.get('judge_issue_rows')}")
    print(f"missing_likert_score_rows: {quality.get('missing_likert_score_rows')}")

    benchmark_examples = quality.get("benchmark_error_examples") or []
    if benchmark_examples:
        print("benchmark_api_error_examples:")
        for item in benchmark_examples[:5]:
            print(
                "  - "
                + " | ".join(
                    [
                        f"model={item.get('model', '')}",
                        f"method={item.get('method', '')}",
                        f"id={item.get('question_id', '')}",
                        f"run={item.get('run_number', '')}",
                        f"error={item.get('error', '')}",
                    ]
                )
            )

    judge_examples = quality.get("judge_issue_examples") or []
    if judge_examples:
        print("judge_issue_examples:")
        for item in judge_examples[:5]:
            print(
                "  - "
                + " | ".join(
                    [
                        f"model={item.get('Model', '')}",
                        f"method={item.get('Method', '')}",
                        f"id={item.get('ID', '')}",
                        f"run={item.get('Run_ID', '')}",
                        f"reason={item.get('Judge_Begruendung', '')}",
                    ]
                )
            )
    print("Empfehlung: denselben run_uuid erneut starten bzw. Judge stabiler/drosselter ausfuehren.")


def _quality_issue_count(summary):
    quality = summary.get("data_quality") or {}
    if not isinstance(quality, dict) or not quality.get("stats_may_be_affected"):
        return 0
    judge_examples = quality.get("judge_issue_examples") or []
    if judge_examples and not quality.get("missing_benchmark_rows") and not quality.get("benchmark_api_error_rows"):
        unique_judge_rows = {
            (
                str(item.get("Model", "")),
                str(item.get("Method", "")),
                str(item.get("ID", "")),
                str(item.get("Run_ID", "")),
            )
            for item in judge_examples
        }
        if unique_judge_rows:
            return len(unique_judge_rows)
    keys = [
        "missing_benchmark_rows",
        "benchmark_api_error_rows",
        "judge_issue_rows",
        "missing_likert_score_rows",
    ]
    total = 0
    for key in keys:
        try:
            total += int(quality.get(key) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _run_pipeline_with_auto_resume(
    *,
    run_cfg,
    cfg,
    judge_prompt_template,
    max_attempts,
    sleep_seconds,
):
    summary = None
    run_uuid = cfg["run_uuid"] or default_run_uuid()
    previous_issue_count = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(
                f"Auto-resume attempt {attempt}/{max_attempts} "
                f"for run_uuid={run_uuid}: offene API/Judge-Probleme werden erneut versucht."
            )
        try:
            summary = run_end_to_end(
                run_config=run_cfg,
                evaluated_output_file=cfg["evaluated_output"],
                judge_model=cfg["judge_model"],
                judge_prompt_version=cfg["judge_prompt_version"],
                judge_prompt_template=judge_prompt_template,
                judge_throttle_seconds=float(cfg.get("judge_throttle_seconds", 0.0)),
                judge_max_retries=int(cfg.get("judge_max_retries", 2)),
                judge_parallel_workers=int(cfg.get("judge_parallel_workers", 1)),
                baseline_method=cfg["baseline_method"],
                time_cost_factor=float(cfg["time_cost_factor"]),
                runs_root=cfg["runs_root"],
                run_uuid=run_uuid,
                enable_overconfidence=bool(cfg.get("enable_overconfidence", False)),
                enable_visualizations=bool(cfg.get("enable_visualizations", False)),
                enable_robustness=bool(cfg.get("enable_robustness", False)),
            )
        except RuntimeError as exc:
            if "Stopped benchmark" not in str(exc) or attempt >= max_attempts:
                raise
            print(
                "Auto-resume: Benchmark wurde wegen API-Fehlern unterbrochen. "
                f"run_uuid={run_uuid} wird nach {sleep_seconds:g}s fortgesetzt."
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue
        run_uuid = summary["run_uuid"]
        issue_count = _quality_issue_count(summary)
        if issue_count <= 0:
            if attempt > 1:
                print(f"Auto-resume abgeschlossen: run_uuid={run_uuid} ist vollstaendig auswertbar.")
            return summary

        if previous_issue_count is not None and issue_count >= previous_issue_count:
            print(
                "Auto-resume Hinweis: Die Anzahl offener Probleme ist nicht gesunken "
                f"({previous_issue_count} -> {issue_count})."
            )
        previous_issue_count = issue_count
        if attempt < max_attempts:
            print(
                f"Auto-resume: {issue_count} offene Datenqualitaetsprobleme erkannt. "
                f"Naechster Versuch nach {sleep_seconds:g}s."
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    assert summary is not None
    print(
        "Auto-resume beendet: maximale Versuchszahl erreicht. "
        "Die finale Statistik wurde fuer den letzten Stand erzeugt; pruefe data_quality_status."
    )
    return summary


def main():
    args = parse_args()
    if args.init_master_config:
        create_master_config_template(args.master_config)
        print(f"Master config template created: {args.master_config}")
        return

    cfg = _resolve_config(args)
    if bool(cfg.get("auto_convert_questions", False)):
        excel_path = str(cfg.get("catalog_excel", "")).strip()
        json_path = str(cfg.get("input_file", "")).strip()
        if excel_path and os.path.exists(excel_path):
            excel_to_json(excel_path=excel_path, json_path=json_path)
        else:
            print(f"Auto-convert skipped: catalog_excel not found ({excel_path})")

    judge_prompt_template = None
    judge_prompt_path = str(cfg.get("judge_prompt_path", "")).strip()
    if judge_prompt_path and Path(judge_prompt_path).exists():
        judge_prompt_template = Path(judge_prompt_path).read_text(encoding="utf-8")
        print(f"Judge prompt file: CUSTOM ({judge_prompt_path})")
    elif judge_prompt_path:
        print(f"Judge prompt file: DEFAULT (missing custom path: {judge_prompt_path})")
    else:
        print("Judge prompt file: DEFAULT (no custom path)")

    run_cfg = RunConfig(
        question_file=cfg["input_file"],
        results_file=cfg["benchmark_output"],
        methods=cfg["methods"],
        models=cfg["models"],
        num_runs=int(cfg["num_runs"]),
        seed=int(cfg["seed"]),
        temperature=float(cfg["temperature"]),
        use_rag=bool(cfg["use_rag"]),
        throttle_seconds=float(cfg["throttle_seconds"]),
        max_retries=int(cfg["max_retries"]),
        model_temperatures=cfg.get("model_temperatures", {}),
        default_max_words=cfg.get("default_max_words"),
        auto_reindex=bool(cfg.get("auto_reindex", True)),
        rag_chunk_size=int(cfg.get("rag_chunk_size", 400)),
        rag_chunk_overlap=int(cfg.get("rag_chunk_overlap", 70)),
        max_rag_context_chars=int(cfg.get("max_rag_context_chars", 6000)),
        max_prompt_chars=int(cfg.get("max_prompt_chars", 14000)),
        slow_call_seconds=float(cfg.get("slow_call_seconds", 10.0)),
        max_consecutive_api_fails=int(cfg.get("max_consecutive_api_fails", 10)),
        parallel_workers=int(cfg.get("parallel_workers", 1)),
        prompt_zero_shot_path=str(cfg.get("prompt_zero_shot_path", "./prompts/zero_shot.txt")),
        prompt_advanced_path=str(cfg.get("prompt_advanced_path", "./prompts/advanced_prompt.txt")),
        prompt_rag_path=str(cfg.get("prompt_rag_path", "./prompts/rag_prompt.txt")),
    )
    auto_resume_until_complete = bool(cfg.get("auto_resume_until_complete", True))
    max_auto_resume_attempts = max(1, int(cfg.get("max_auto_resume_attempts", 5)))
    auto_resume_sleep_seconds = max(0.0, float(cfg.get("auto_resume_sleep_seconds", 30.0)))
    summary = _run_pipeline_with_auto_resume(
        run_cfg=run_cfg,
        cfg=cfg,
        judge_prompt_template=judge_prompt_template,
        max_attempts=max_auto_resume_attempts if auto_resume_until_complete else 1,
        sleep_seconds=auto_resume_sleep_seconds,
    )
    print("Pipeline completed.")
    print(f"master_config: {args.master_config}")
    print(f"run_uuid: {summary['run_uuid']}")
    print(f"run_dir: {summary['run_dir']}")
    print(f"benchmark_output: {summary['benchmark_output']}")
    print(f"evaluation_output: {summary['evaluation_output']}")
    print(f"stats_report_json: {summary['stats_report_json']}")
    print(f"metrics_summary_csv: {summary['metrics_summary_csv']}")
    print(f"summary_main_tests_csv: {summary.get('summary_main_tests_csv')}")
    print(f"summary_trap_questions_csv: {summary.get('summary_trap_questions_csv')}")
    print(f"summary_robustness_csv: {summary.get('summary_robustness_csv')}")
    print(f"summary_overconfidence_csv: {summary.get('summary_overconfidence_csv')}")
    print(f"summary_tasktype_deltas_csv: {summary.get('summary_tasktype_deltas_csv')}")
    print(f"figures_dir: {summary['figures_dir']}")
    print(f"figure_files: {len(summary['figure_files'])}")
    _print_data_quality_summary(summary)


if __name__ == "__main__":
    main()
