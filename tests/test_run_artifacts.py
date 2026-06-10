from src.benchmarking.logging_utils import init_run_artifacts


def test_init_run_artifacts_reuses_run_uuid():
    artifacts_a = init_run_artifacts("./eval_data/runs", run_uuid="test-run-123")
    artifacts_b = init_run_artifacts("./eval_data/runs", run_uuid="test-run-123")
    assert artifacts_a["run_uuid"] == "test-run-123"
    assert artifacts_b["run_uuid"] == "test-run-123"
    assert artifacts_a["run_dir"] == artifacts_b["run_dir"]

