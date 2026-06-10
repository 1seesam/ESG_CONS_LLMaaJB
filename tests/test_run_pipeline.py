import run_pipeline

from src.benchmarking.config_schema import RunConfig


def test_auto_resume_forwards_enable_visualizations_flag(monkeypatch):
    captured = {}

    def fake_run_end_to_end(**kwargs):
        captured["enable_visualizations"] = kwargs["enable_visualizations"]
        return {
            "run_uuid": "run-x",
            "run_dir": "./tmp/run-x",
            "benchmark_output": "./tmp/benchmark.xlsx",
            "evaluation_output": "./tmp/evaluated.xlsx",
            "stats_report_json": "./tmp/stats.json",
            "metrics_summary_csv": "./tmp/metrics.csv",
            "figure_files": [],
        }

    monkeypatch.setattr(run_pipeline, "run_end_to_end", fake_run_end_to_end)

    cfg = {
        "evaluated_output": "./tmp/evaluated.xlsx",
        "judge_model": "judge-model",
        "judge_prompt_version": "trap_v2",
        "judge_throttle_seconds": 0.0,
        "judge_max_retries": 2,
        "judge_parallel_workers": 1,
        "baseline_method": "Zero-Shot",
        "time_cost_factor": 0.002,
        "runs_root": "./tmp/runs",
        "run_uuid": "run-x",
        "enable_overconfidence": False,
        "enable_visualizations": False,
        "enable_robustness": False,
    }

    run_pipeline._run_pipeline_with_auto_resume(
        run_cfg=RunConfig(question_file="./eval_data/questions.json"),
        cfg=cfg,
        judge_prompt_template=None,
        max_attempts=1,
        sleep_seconds=0.0,
    )

    assert captured["enable_visualizations"] is False
