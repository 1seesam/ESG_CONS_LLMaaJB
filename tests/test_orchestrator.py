from pathlib import Path

import pandas as pd

from src.benchmarking.config_schema import RunConfig
from src.benchmarking.orchestrator import run_end_to_end


def test_run_end_to_end_summary(monkeypatch):
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_orchestrator"
    run_dir = base / "eval_data" / "runs" / "run-x"
    run_dir.mkdir(parents=True, exist_ok=True)

    def fake_benchmark(run_config, paths=None, run_uuid=None):
        df = pd.DataFrame([{"a": 1}])
        artifacts = {
            "run_uuid": run_uuid or "run-x",
            "run_dir": str(run_dir),
            "raw_results_jsonl": str(run_dir / "raw_results.jsonl"),
        }
        return df, artifacts

    def fake_eval(**kwargs):
        output_file = Path(kwargs["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"Model": "m1", "Method": "Zero-Shot"}]).to_excel(output_file, index=False)
        return pd.DataFrame([{"b": 1}]), {
            "evaluated_results_jsonl": str(run_dir / "evaluated_results.jsonl"),
            "run_dir": str(run_dir),
        }

    def fake_analysis(**kwargs):
        return pd.DataFrame([{"Model": "m1", "Method": "Zero-Shot"}]), {
            "stats_report_json": str(run_dir / "stats_report.json"),
            "run_dir": str(run_dir),
        }

    monkeypatch.setattr("src.benchmarking.orchestrator.run_benchmark", fake_benchmark)
    monkeypatch.setattr("src.benchmarking.orchestrator.run_evaluation", fake_eval)
    monkeypatch.setattr("src.benchmarking.orchestrator.run_analysis", fake_analysis)

    cfg = RunConfig(question_file="./eval_data/questions.json")
    summary = run_end_to_end(
        run_config=cfg,
        evaluated_output_file=str(base / "eval_data" / "results_eval.xlsx"),
        runs_root=str(base / "eval_data" / "runs"),
        run_uuid="run-x",
    )
    assert summary["run_uuid"] == "run-x"
    assert summary["run_dir"].endswith("run-x")
    assert Path(summary["pipeline_summary_json"]).exists()
