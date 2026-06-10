import json
import shutil
from pathlib import Path

from src.benchmarking.config_schema import PathsConfig, QuestionItem, RunConfig
from src.benchmarking.io import read_jsonl
from src.benchmarking.pipeline import run_benchmark


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "pytest_pipeline_resume"


def _write_question_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


def test_run_benchmark_persists_checkpoint_before_abort(monkeypatch):
    base = TEST_TMP_ROOT / "checkpoint_abort"
    if base.exists():
        shutil.rmtree(base)
    question_file = base / "questions.json"
    results_file = base / "benchmark.xlsx"
    runs_root = base / "runs"
    _write_question_stub(question_file)

    questions = [
        QuestionItem(id="Q1", question="Frage 1"),
        QuestionItem(id="Q2", question="Frage 2"),
    ]
    responses = iter(
        [
            {"content": "ok-1", "input_tokens": 1, "output_tokens": 1, "duration_sec": 0.1, "error": ""},
            {"content": "err-2", "input_tokens": 0, "output_tokens": 0, "duration_sec": 0.0, "error": "api down"},
        ]
    )

    monkeypatch.setattr("src.benchmarking.pipeline.load_questions", lambda path: questions)
    monkeypatch.setattr("src.benchmarking.pipeline.get_llm", lambda model_name, temperature: object())
    monkeypatch.setattr(
        "src.benchmarking.pipeline.resolve_pricing_for_models",
        lambda **kwargs: ({}, {"source": "test"}),
    )
    monkeypatch.setattr(
        "src.benchmarking.pipeline.load_prompt_templates",
        lambda mapping: {key: "{question}" for key in mapping},
    )
    monkeypatch.setattr(
        "src.benchmarking.pipeline.invoke_with_retry",
        lambda **kwargs: next(responses),
    )

    cfg = RunConfig(
        question_file=str(question_file),
        results_file=str(results_file),
        methods=["Zero-Shot"],
        models=["model-a"],
        num_runs=1,
        use_rag=False,
        max_consecutive_api_fails=0,
    )

    try:
        run_benchmark(cfg, run_uuid="resume-run", paths=PathsConfig(runs_root=str(runs_root)))
    except RuntimeError as exc:
        assert "Resume by rerunning with run_uuid=resume-run" in str(exc)
    else:
        raise AssertionError("Expected benchmark abort due to consecutive API failures.")

    raw_rows = read_jsonl(str(runs_root / "resume-run" / "raw_results.jsonl"))
    assert len(raw_rows) == 2
    assert raw_rows[0]["question_id"] == "Q1"
    assert raw_rows[0]["error"] == ""
    assert raw_rows[1]["question_id"] == "Q2"
    assert raw_rows[1]["error"] == "api down"
    assert results_file.exists()


def test_run_benchmark_resumes_only_failed_tasks(monkeypatch):
    base = TEST_TMP_ROOT / "resume_failed_only"
    if base.exists():
        shutil.rmtree(base)
    question_file = base / "questions.json"
    results_file = base / "benchmark.xlsx"
    runs_root = base / "runs"
    _write_question_stub(question_file)

    questions = [
        QuestionItem(id="Q1", question="Frage 1"),
        QuestionItem(id="Q2", question="Frage 2"),
    ]
    call_log: list[str] = []
    responses = iter(
        [
            {"content": "ok-2", "input_tokens": 2, "output_tokens": 1, "duration_sec": 0.2, "error": ""},
        ]
    )

    run_dir = runs_root / "resume-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_results.jsonl"
    raw_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_uuid": "resume-run",
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        "config_hash": "hash",
                        "run_number": 1,
                        "model": "model-a",
                        "method": "Zero-Shot",
                        "question_id": "Q1",
                        "question": "Frage 1",
                        "answer_type": "",
                        "llm_answer": "ok-1",
                        "reference_answer": "",
                        "atomic_facts": "[]",
                        "document_name": "n/a",
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 0.1,
                        "retrieval_sources": "",
                        "error": "",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "run_uuid": "resume-run",
                        "timestamp_utc": "2026-01-01T00:00:01+00:00",
                        "config_hash": "hash",
                        "run_number": 1,
                        "model": "model-a",
                        "method": "Zero-Shot",
                        "question_id": "Q2",
                        "question": "Frage 2",
                        "answer_type": "",
                        "llm_answer": "err-2",
                        "reference_answer": "",
                        "atomic_facts": "[]",
                        "document_name": "n/a",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 0.0,
                        "retrieval_sources": "",
                        "error": "api down",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("src.benchmarking.pipeline.load_questions", lambda path: questions)
    monkeypatch.setattr("src.benchmarking.pipeline.get_llm", lambda model_name, temperature: object())
    monkeypatch.setattr(
        "src.benchmarking.pipeline.resolve_pricing_for_models",
        lambda **kwargs: ({}, {"source": "test"}),
    )
    monkeypatch.setattr(
        "src.benchmarking.pipeline.load_prompt_templates",
        lambda mapping: {key: "{question}" for key in mapping},
    )

    def fake_invoke(**kwargs):
        call_log.append(kwargs["prompt"])
        return next(responses)

    monkeypatch.setattr("src.benchmarking.pipeline.invoke_with_retry", fake_invoke)

    cfg = RunConfig(
        question_file=str(question_file),
        results_file=str(results_file),
        methods=["Zero-Shot"],
        models=["model-a"],
        num_runs=1,
        use_rag=False,
        max_consecutive_api_fails=0,
    )

    df, artifacts = run_benchmark(cfg, run_uuid="resume-run", paths=PathsConfig(runs_root=str(runs_root)))

    assert len(call_log) == 1
    assert len(df) == 2
    assert set(df["question_id"]) == {"Q1", "Q2"}
    q2_row = df[df["question_id"] == "Q2"].iloc[0]
    assert q2_row["error"] == ""
    assert q2_row["llm_answer"] == "ok-2"

    raw_rows = read_jsonl(artifacts["raw_results_jsonl"])
    assert len(raw_rows) == 2
    assert {row["question_id"] for row in raw_rows} == {"Q1", "Q2"}
    assert results_file.exists()
