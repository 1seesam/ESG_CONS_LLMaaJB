from pathlib import Path

import pandas as pd

from src.benchmarking.analysis_pipeline import run_analysis
from src.benchmarking.statistics import (
    THESIS_MAIN_COLUMNS,
    THESIS_OVERCONFIDENCE_COLUMNS,
    THESIS_ROBUSTNESS_COLUMNS,
    THESIS_TRAP_COLUMNS,
    build_thesis_main_tests,
    build_thesis_trap_summary,
)


def _row(
    model: str,
    method: str,
    qid: str,
    run_id: int,
    support_rate: float,
    contradiction_rate: float,
    *,
    is_trap_question: int = 0,
    trap_pass: float = float("nan"),
    confidence_mean: float = 1.0,
):
    atomic_precision = max(0.0, 1.0 - contradiction_rate)
    return {
        "Model": model,
        "Method": method,
        "ID": qid,
        "Run_ID": run_id,
        "Likert_Score": support_rate * 4.0,
        "Atomic_Coverage_Score": support_rate,
        "Support_Rate": support_rate,
        "Atomic_Precision": atomic_precision,
        "Hallucination_Rate": 1.0 - atomic_precision,
        "Contradiction_Rate": contradiction_rate,
        "Not_In_Scope_Rate": 0.0,
        "Trap_Pass": trap_pass,
        "Trap_Hallucination": (1.0 - trap_pass) if pd.notna(trap_pass) else float("nan"),
        "Is_Trap_Question": is_trap_question,
        "Confidence_Mean": confidence_mean,
        "Confidence_Found_Flag": 1,
        "KI_Antwort": f"{method} answer {qid} run {run_id}",
        "Cost_USD": 0.0,
        "Dauer_sek": 1.0,
    }


def test_thesis_main_tests_have_fixed_shape_and_metric_direction():
    rows = []
    for qid, zs_acc, adv_acc, rag_acc, zs_hall, adv_hall, rag_hall in [
        ("q1", 0.10, 0.40, 0.70, 0.60, 0.20, 0.10),
        ("q2", 0.20, 0.50, 0.80, 0.50, 0.20, 0.00),
        ("q3", 0.30, 0.60, 0.90, 0.40, 0.10, 0.10),
        ("q4", 0.40, 0.50, 1.00, 0.40, 0.20, 0.00),
    ]:
        rows.append(_row("m1", "Zero-Shot", qid, 1, zs_acc, zs_hall))
        rows.append(_row("m1", "Advanced-Prompt", qid, 1, adv_acc, adv_hall))
        rows.append(_row("m1", "RAG", qid, 1, rag_acc, rag_hall))
    main_df = build_thesis_main_tests(pd.DataFrame(rows))

    assert list(main_df.columns) == THESIS_MAIN_COLUMNS
    assert len(main_df) == 9
    assert set(main_df["comparison"]) == {
        "Zero-Shot vs Advanced Prompting",
        "Zero-Shot vs RAG",
        "Advanced Prompting vs RAG",
    }

    atomic_df = main_df[main_df["metric"] == "atomic_coverage_score"].set_index("comparison")

    assert atomic_df.loc["Zero-Shot vs Advanced Prompting", "direction"] == "favours_Advanced Prompting"
    assert atomic_df.loc["Zero-Shot vs RAG", "direction"] == "favours_RAG"
    assert atomic_df.loc["Advanced Prompting vs RAG", "direction"] == "favours_RAG"
    assert set(main_df["metric"]) == {"atomic_coverage_score", "atomic_precision", "hallucination_rate"}
    assert atomic_df.loc["Zero-Shot vs RAG", "wilcoxon_statistic"] == 0.0
    assert pd.notna(atomic_df.loc["Zero-Shot vs RAG", "p_value_bh"])
    assert atomic_df.loc["Zero-Shot vs RAG", "effect_size_label"] in {"klein", "mittel", "gross"}
    assert "shapiro_p_value" in main_df.columns


def test_trap_questions_are_excluded_from_main_tests_and_exported_separately():
    rows = [
        _row("m1", "Zero-Shot", "q1", 1, 0.2, 0.4),
        _row("m1", "RAG", "q1", 1, 0.8, 0.1),
        _row("m1", "Zero-Shot", "q2", 1, 0.3, 0.3, is_trap_question=1, trap_pass=1.0),
        _row("m1", "RAG", "q2", 1, 0.7, 0.1, is_trap_question=1, trap_pass=0.0),
        _row("m1", "Zero-Shot", "q3", 1, 0.1, 0.5),
        _row("m1", "RAG", "q3", 1, 0.9, 0.0),
    ]
    df = pd.DataFrame(rows)

    main_df = build_thesis_main_tests(df)
    trap_df = build_thesis_trap_summary(df)

    main_row = main_df[
        (main_df["metric"] == "atomic_coverage_score") & (main_df["comparison"] == "Zero-Shot vs RAG")
    ].iloc[0]
    assert main_row["n_pairs"] == 2

    assert list(trap_df.columns) == THESIS_TRAP_COLUMNS
    zero_row = trap_df[trap_df["method"] == "Zero-Shot"].iloc[0]
    rag_row = trap_df[trap_df["method"] == "RAG"].iloc[0]
    assert zero_row["n_trap_questions"] == 1
    assert zero_row["correct_refusal_rate"] == 1.0
    assert zero_row["hallucinated_answer_rate"] == 0.0
    assert rag_row["correct_refusal_rate"] == 0.0
    assert rag_row["hallucinated_answer_rate"] == 1.0


def test_missing_methods_do_not_create_placeholder_rows_in_thesis_exports():
    df = pd.DataFrame(
        [
            _row("m1", "Zero-Shot", "q1", 1, 0.2, 0.4),
            _row("m1", "RAG", "q1", 1, 0.8, 0.1),
            _row("m1", "Zero-Shot", "q2", 1, 0.1, 0.5, is_trap_question=1, trap_pass=1.0),
            _row("m1", "RAG", "q2", 1, 0.7, 0.1, is_trap_question=1, trap_pass=0.0),
            _row("m1", "Zero-Shot", "q3", 1, 0.1, 0.5),
            _row("m1", "RAG", "q3", 1, 0.9, 0.0),
        ]
    )

    main_df = build_thesis_main_tests(df)
    trap_df = build_thesis_trap_summary(df)

    assert main_df["comparison"].tolist() == ["Zero-Shot vs RAG", "Zero-Shot vs RAG", "Zero-Shot vs RAG"]
    assert set(main_df["metric"]) == {"atomic_coverage_score", "atomic_precision", "hallucination_rate"}
    assert "Advanced Prompting" not in trap_df["method"].tolist()


def test_run_analysis_writes_required_and_optional_thesis_exports():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_analysis_exports"
    base.mkdir(parents=True, exist_ok=True)
    input_file = base / "evaluated.xlsx"
    runs_root = base / "runs"

    df = pd.DataFrame(
        [
            _row("m1", "Zero-Shot", "q1", 1, 0.2, 0.5, confidence_mean=0.0),
            _row("m1", "Advanced-Prompt", "q1", 1, 0.4, 0.3, confidence_mean=1.0),
            _row("m1", "RAG", "q1", 1, 0.8, 0.1, confidence_mean=2.0),
            _row("m1", "Zero-Shot", "q2", 1, 0.1, 0.6, confidence_mean=0.0),
            _row("m1", "Advanced-Prompt", "q2", 1, 0.5, 0.2, confidence_mean=1.0),
            _row("m1", "RAG", "q2", 1, 0.9, 0.0, confidence_mean=2.0),
            _row("m1", "Zero-Shot", "q1", 2, 0.3, 0.5, confidence_mean=0.0),
            _row("m1", "Advanced-Prompt", "q1", 2, 0.5, 0.3, confidence_mean=1.0),
            _row("m1", "RAG", "q1", 2, 0.9, 0.1, confidence_mean=2.0),
            _row("m1", "Zero-Shot", "q2", 2, 0.2, 0.6, confidence_mean=0.0),
            _row("m1", "Advanced-Prompt", "q2", 2, 0.6, 0.2, confidence_mean=1.0),
            _row("m1", "RAG", "q2", 2, 1.0, 0.0, confidence_mean=2.0),
        ]
    )
    df.to_excel(input_file, index=False)

    _, artifacts = run_analysis(
        input_file=str(input_file),
        runs_root=str(runs_root),
        run_uuid="run-a",
        enable_overconfidence=False,
        enable_robustness=False,
    )
    assert Path(artifacts["summary_main_tests_csv"]).exists()
    assert Path(artifacts["summary_trap_questions_csv"]).exists()
    assert not Path(artifacts["summary_robustness_csv"]).exists()
    assert not Path(artifacts["summary_overconfidence_csv"]).exists()

    _, artifacts = run_analysis(
        input_file=str(input_file),
        runs_root=str(runs_root),
        run_uuid="run-b",
        enable_overconfidence=True,
        enable_robustness=True,
    )
    robustness_df = pd.read_csv(artifacts["summary_robustness_csv"])
    overconfidence_df = pd.read_csv(artifacts["summary_overconfidence_csv"])

    assert list(robustness_df.columns) == THESIS_ROBUSTNESS_COLUMNS
    assert list(overconfidence_df.columns) == THESIS_OVERCONFIDENCE_COLUMNS
