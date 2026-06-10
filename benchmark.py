import argparse
import os
import subprocess
import sys
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

    if os.environ.get("THESIS_BENCHMARK_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["THESIS_BENCHMARK_REEXEC"] = "1"
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

from src.benchmarking.config_schema import PathsConfig, RunConfig
from src.benchmarking.master_config import create_master_config_template, load_master_config
from src.benchmarking.pipeline import run_benchmark
from src.convert_questions import excel_to_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark-only runner with optional Excel->JSON conversion."
    )
    parser.add_argument("--master-config", default="./eval_data/master_config.xlsx")
    parser.add_argument(
        "--init-master-config",
        action="store_true",
        help="Create a template master config Excel and exit.",
    )
    parser.add_argument("--catalog-excel", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--benchmark-output", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--run-uuid", default=None)
    parser.add_argument("--auto-convert", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--use-rag", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--throttle-seconds", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--default-max-words", type=int, default=None)
    parser.add_argument("--auto-reindex", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rag-chunk-size", type=int, default=None)
    parser.add_argument("--rag-chunk-overlap", type=int, default=None)
    parser.add_argument("--max-rag-context-chars", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--slow-call-seconds", type=float, default=None)
    parser.add_argument("--max-consecutive-api-fails", type=int, default=None)
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--prompt-zero-shot-path", default=None)
    parser.add_argument("--prompt-advanced-path", default=None)
    parser.add_argument("--prompt-rag-path", default=None)
    return parser.parse_args()


def _resolve_config(args):
    cfg = load_master_config(args.master_config)
    if args.catalog_excel is not None:
        cfg["catalog_excel"] = args.catalog_excel
    if args.input_file is not None:
        cfg["input_file"] = args.input_file
    if args.benchmark_output is not None:
        cfg["benchmark_output"] = args.benchmark_output
    if args.runs_root is not None:
        cfg["runs_root"] = args.runs_root
    if args.run_uuid is not None:
        cfg["run_uuid"] = args.run_uuid
    if args.auto_convert is not None:
        cfg["auto_convert_questions"] = bool(args.auto_convert)

    overrides = {
        "num_runs": args.num_runs,
        "seed": args.seed,
        "temperature": args.temperature,
        "methods": args.methods,
        "models": args.models,
        "use_rag": args.use_rag,
        "throttle_seconds": args.throttle_seconds,
        "max_retries": args.max_retries,
        "default_max_words": args.default_max_words,
        "auto_reindex": args.auto_reindex,
        "rag_chunk_size": args.rag_chunk_size,
        "rag_chunk_overlap": args.rag_chunk_overlap,
        "max_rag_context_chars": args.max_rag_context_chars,
        "max_prompt_chars": args.max_prompt_chars,
        "slow_call_seconds": args.slow_call_seconds,
        "max_consecutive_api_fails": args.max_consecutive_api_fails,
        "parallel_workers": args.parallel_workers,
        "prompt_zero_shot_path": args.prompt_zero_shot_path,
        "prompt_advanced_path": args.prompt_advanced_path,
        "prompt_rag_path": args.prompt_rag_path,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def main():
    args = parse_args()
    if args.init_master_config:
        create_master_config_template(args.master_config)
        print(f"Master config template created: {args.master_config}")
        return

    cfg = _resolve_config(args)
    if bool(cfg.get("auto_convert_questions", True)):
        excel_path = str(cfg.get("catalog_excel", "")).strip()
        json_path = str(cfg.get("input_file", os.getenv("QUESTION_FILE", "./eval_data/questions.json"))).strip()
        if excel_path and os.path.exists(excel_path):
            excel_to_json(excel_path=excel_path, json_path=json_path)
        else:
            print(f"Auto-convert skipped: catalog_excel not found ({excel_path})")

    config = RunConfig(
        question_file=str(cfg.get("input_file", os.getenv("QUESTION_FILE", "./eval_data/questions.json"))).strip(),
        results_file=str(cfg.get("benchmark_output", "./eval_data/results_benchmark_final.xlsx")).strip(),
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
    run_uuid = str(cfg.get("run_uuid", "") or "").strip() or None
    df, artifacts = run_benchmark(
        config,
        paths=PathsConfig(runs_root=str(cfg.get("runs_root", "./eval_data/runs")).strip()),
        run_uuid=run_uuid,
    )
    if df.empty:
        print("No benchmark results were generated.")
        return

    print(f"Benchmark finished: {len(df)} rows")
    print(f"master_config: {args.master_config}")
    print(f"results: {config.results_file}")
    print(f"run_artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()
