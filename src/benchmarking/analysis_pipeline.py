from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd

from .io import read_table, write_json
from .logging_utils import init_run_artifacts
from .statistics import (
    build_stats_tables,
    build_thesis_main_tests,
    build_thesis_overconfidence_summary,
    build_thesis_robustness_summary,
    build_thesis_trap_summary,
)


def run_analysis(
    input_file: str,
    baseline_method: str = "Zero-Shot",
    runs_root: str = "./eval_data/runs",
    time_cost_factor: float = 0.002,
    run_uuid: str | None = None,
    enable_overconfidence: bool = False,
    enable_robustness: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = read_table(input_file)
    stats_df = build_stats_tables(
        df,
        baseline_method=baseline_method,
        time_cost_factor=time_cost_factor,
    )
    main_tests_df = build_thesis_main_tests(df)
    trap_summary_df = build_thesis_trap_summary(df)
    robustness_df = (
        build_thesis_robustness_summary(df, main_tests_df) if enable_robustness else pd.DataFrame()
    )
    overconfidence_df = (
        build_thesis_overconfidence_summary(df) if enable_overconfidence else pd.DataFrame()
    )

    artifacts = init_run_artifacts(runs_root, run_uuid=run_uuid)
    report = stats_df.to_dict(orient="records")
    write_json(artifacts["stats_report_json"], report)
    main_tests_df.to_csv(artifacts["summary_main_tests_csv"], index=False)
    trap_summary_df.to_csv(artifacts["summary_trap_questions_csv"], index=False)
    if enable_robustness:
        robustness_df.to_csv(artifacts["summary_robustness_csv"], index=False)
    if enable_overconfidence and not overconfidence_df.empty:
        overconfidence_df.to_csv(artifacts["summary_overconfidence_csv"], index=False)

    stage_cfg_path = os.path.join(artifacts["run_dir"], "analysis.config.json")
    write_json(
        stage_cfg_path,
        {
            "stage": "analysis",
            "input_file": input_file,
            "baseline_method": baseline_method,
            "time_cost_factor": time_cost_factor,
            "primary_inference_test": "wilcoxon_signed_rank",
            "primary_inference_unit": "task",
            "stability_check": "run_level descriptive stability check",
            "multiple_testing": "benjamini_hochberg_per_model",
            "confidence_normalization": "1 - (confidence_mean / 3)",
            "score_normalization": "support_rate",
            "overconfidence_formula": "confidence_norm * (1 - support_rate)",
            "thesis_exports": {
                "summary_main_tests_csv": artifacts["summary_main_tests_csv"],
                "summary_trap_questions_csv": artifacts["summary_trap_questions_csv"],
                "summary_robustness_csv": artifacts["summary_robustness_csv"] if enable_robustness else None,
                "summary_overconfidence_csv": artifacts["summary_overconfidence_csv"] if enable_overconfidence else None,
            },
            "optional_features": {
                "enable_overconfidence": bool(enable_overconfidence),
                "enable_robustness": bool(enable_robustness),
            },
            "methodology_note": (
                "Primary inference is performed only on non-trap tasks at task level (Model+ID) "
                "using a paired Wilcoxon Signed-Rank Test on atomic coverage score. "
                "Trap questions are analysed separately via correct refusal and hallucinated answer rates. "
                "Confidence is parsed only from explicit #0-#3 fact markers. Costs and timing are descriptive only."
            ),
        },
    )
    return stats_df, artifacts

