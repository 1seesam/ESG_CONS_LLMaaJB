import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_project_cwd() -> None:
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass

from src.benchmarking.analysis_pipeline import run_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evaluated benchmark results.")
    parser.add_argument("--input-file", default="./eval_data/results_evaluated_scientific.xlsx")
    parser.add_argument("--baseline-method", default="Zero-Shot")
    parser.add_argument("--runs-root", default="./eval_data/runs")
    parser.add_argument("--time-cost-factor", type=float, default=0.002)
    parser.add_argument("--run-uuid", default=None, help="Optional existing run UUID to reuse.")
    return parser.parse_args()


def main():
    args = parse_args()
    stats_df, artifacts = run_analysis(
        input_file=args.input_file,
        baseline_method=args.baseline_method,
        runs_root=args.runs_root,
        time_cost_factor=args.time_cost_factor,
        run_uuid=args.run_uuid,
    )
    print("Final statistics table:")
    print(
        stats_df[
            [
                "Model",
                "Method",
                "N",
                "Mean_Score",
                "Critical_Fail",
                "Std_Dev",
                "Semantic_Consistency",
                "p_value",
                "q_value",
                "effect_size",
                "Biz Value Index",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"Stats report: {artifacts['stats_report_json']}")


if __name__ == "__main__":
    _ensure_project_cwd()
    main()

