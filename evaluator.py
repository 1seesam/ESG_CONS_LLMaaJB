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

    if os.environ.get("THESIS_EVALUATOR_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["THESIS_EVALUATOR_REEXEC"] = "1"
    completed = subprocess.run([str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env, cwd=str(PROJECT_ROOT))
    raise SystemExit(completed.returncode)


def _ensure_project_cwd() -> None:
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass


if __name__ == "__main__":
    _ensure_project_cwd()
    _reexec_with_project_venv_if_needed()

from src.benchmarking.evaluation_pipeline import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark outputs.")
    parser.add_argument("--input-file", default="./eval_data/results_benchmark_final.xlsx")
    parser.add_argument("--output-file", default="./eval_data/results_evaluated_scientific.xlsx")
    parser.add_argument("--judge-model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge-prompt-version", default="trap_v2")
    parser.add_argument("--judge-throttle-seconds", type=float, default=0.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-parallel-workers", type=int, default=1)
    parser.add_argument("--runs-root", default="./eval_data/runs")
    parser.add_argument("--run-uuid", default=None, help="Optional existing run UUID to reuse.")
    return parser.parse_args()


def main():
    args = parse_args()
    df, artifacts = run_evaluation(
        input_file=args.input_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        judge_prompt_version=args.judge_prompt_version,
        judge_throttle_seconds=args.judge_throttle_seconds,
        judge_max_retries=args.judge_max_retries,
        judge_parallel_workers=args.judge_parallel_workers,
        runs_root=args.runs_root,
        run_uuid=args.run_uuid,
    )
    print(f"Evaluation finished: {len(df)} rows")
    print(f"Evaluated file: {args.output_file}")
    print(f"Run artifacts: {artifacts['run_dir']}")


if __name__ == "__main__":
    main()

