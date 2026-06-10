import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.convert_questions import excel_to_json
from src.benchmarking.visualization import (
    _build_run_variance_summary,
    _build_tasktype_delta_frame,
    _build_tasktype_delta_summary,
    _normalize_task_type,
    generate_run_figures,
)


def test_excel_to_json_preserves_task_type_raw():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_tasktype_convert"
    base.mkdir(parents=True, exist_ok=True)
    excel_path = base / "catalog.xlsx"
    json_path = base / "questions.json"

    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Kategorie": "Regulatorik",
                "Aufgabentyp": "Wissen",
                "Frage": "Was gilt?",
                "Referenzantwort": "Etwas",
                "Schwierigkeitslevel": 2,
                "Antwortoptionen": "Atomic Facts",
            }
        ]
    )
    df.to_excel(excel_path, index=False)

    excel_to_json(excel_path=str(excel_path), json_path=str(json_path))

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload[0]["task_type_raw"] == "Wissen"


def test_task_type_normalization_and_delta_aggregation():
    assert _normalize_task_type("Wissen ") == "Wissensbasiert"
    assert _normalize_task_type("Analytische Frage") == "Analytisch / Reasoning"
    assert _normalize_task_type("Kalkulation") == "Rechen-/deterministisch"
    assert _normalize_task_type("sonstiges") == "Unklar/sonstige"

    df = pd.DataFrame(
        [
            {
                "ID": "q1",
                "Model": "m1",
                "Method": "Zero-Shot",
                "Support_Rate": 0.2,
                "Is_Trap_Question": 0,
                "Aufgabentyp": "Wissensbasiert",
            },
            {
                "ID": "q1",
                "Model": "m1",
                "Method": "RAG",
                "Support_Rate": 0.4,
                "Is_Trap_Question": 0,
                "Aufgabentyp": "Wissensbasiert",
            },
            {
                "ID": "q2",
                "Model": "m1",
                "Method": "Zero-Shot",
                "Support_Rate": 1.0,
                "Is_Trap_Question": 1,
                "Aufgabentyp": "Wissensbasiert",
            },
            {
                "ID": "q2",
                "Model": "m1",
                "Method": "RAG",
                "Support_Rate": 0.2,
                "Is_Trap_Question": 0,
                "Aufgabentyp": "Wissensbasiert",
            },
            {
                "ID": "q3",
                "Model": "m2",
                "Method": "Zero-Shot",
                "Support_Rate": 0.5,
                "Is_Trap_Question": 0,
                "Aufgabentyp": "Wissensbasiert",
            },
            {
                "ID": "q3",
                "Model": "m2",
                "Method": "RAG",
                "Support_Rate": 0.7,
                "Is_Trap_Question": 0,
                "Aufgabentyp": "Wissensbasiert",
            },
        ]
    )

    deltas = _build_tasktype_delta_frame(df)
    precision_df = df.copy()
    precision_df["Atomic_Precision"] = [0.4, 0.7, 1.0, 0.2, 0.5, 0.8]
    precision_deltas = _build_tasktype_delta_frame(precision_df, metric_key="atomic_precision")
    hallucination_df = precision_df.copy()
    hallucination_df["Hallucination_Rate"] = 1.0 - hallucination_df["Atomic_Precision"]
    hallucination_deltas = _build_tasktype_delta_frame(hallucination_df, metric_key="hallucination_rate")
    summary = _build_tasktype_delta_summary(df)

    assert set(deltas["comparison"]) == {"RAG - Zero-Shot"}
    assert len(deltas) == 2
    assert abs(float(deltas.iloc[0]["delta_atomic_accuracy"]) - 0.2) < 1e-9
    assert abs(float(precision_deltas.iloc[0]["delta_atomic_precision"]) - 0.3) < 1e-9
    assert abs(float(hallucination_deltas.iloc[0]["delta_hallucination_rate"]) - (-0.3)) < 1e-9
    row = summary.iloc[0]
    assert row["task_type"] == "Wissensbasiert"
    assert row["comparison"] == "RAG - Zero-Shot"
    assert row["n_pairs"] == 2
    assert abs(float(row["median_delta_atomic_accuracy"]) - 0.2) < 1e-9


def test_generate_run_figures_creates_task_type_figures():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_tasktype_figures"
    base.mkdir(parents=True, exist_ok=True)
    question_file = base / "questions.json"
    outdir = base / "figures"

    questions = [
        {
            "id": "q1",
            "category": "Regulatorik",
            "difficulty": "1",
            "task_type_raw": "Wissen",
        },
        {
            "id": "q2",
            "category": "Gebäudetechnik",
            "difficulty": "2",
            "task_type_raw": "Kalkulation",
        },
        {
            "id": "q3",
            "category": "ESG-Reporting",
            "difficulty": "3",
            "task_type_raw": "Analyse",
        },
    ]
    with question_file.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False)

    evaluated_df = pd.DataFrame(
        [
            {"ID": "q1", "Model": "m1", "Method": "Zero-Shot", "Atomic_Coverage_Score": 0.2, "Support_Rate": 0.2, "Atomic_Precision": 0.6, "Hallucination_Rate": 0.4, "Contradiction_Rate": 0.2, "Not_In_Scope_Rate": 0.6, "Likert_Score": 1, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q2", "Model": "m1", "Method": "Zero-Shot", "Atomic_Coverage_Score": 0.5, "Support_Rate": 0.5, "Atomic_Precision": 0.7, "Hallucination_Rate": 0.3, "Contradiction_Rate": 0.1, "Not_In_Scope_Rate": 0.4, "Likert_Score": 2, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q3", "Model": "m1", "Method": "Zero-Shot", "Atomic_Coverage_Score": 0.7, "Support_Rate": 0.7, "Atomic_Precision": 0.8, "Hallucination_Rate": 0.2, "Contradiction_Rate": 0.1, "Not_In_Scope_Rate": 0.2, "Likert_Score": 3, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q1", "Model": "m1", "Method": "Advanced-Prompt", "Atomic_Coverage_Score": 0.6, "Support_Rate": 0.6, "Atomic_Precision": 0.8, "Hallucination_Rate": 0.2, "Contradiction_Rate": 0.1, "Not_In_Scope_Rate": 0.3, "Likert_Score": 3, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q2", "Model": "m1", "Method": "Advanced-Prompt", "Atomic_Coverage_Score": 0.7, "Support_Rate": 0.7, "Atomic_Precision": 0.85, "Hallucination_Rate": 0.15, "Contradiction_Rate": 0.1, "Not_In_Scope_Rate": 0.2, "Likert_Score": 4, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q3", "Model": "m1", "Method": "Advanced-Prompt", "Atomic_Coverage_Score": 0.8, "Support_Rate": 0.8, "Atomic_Precision": 0.9, "Hallucination_Rate": 0.1, "Contradiction_Rate": 0.0, "Not_In_Scope_Rate": 0.2, "Likert_Score": 3, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q1", "Model": "m1", "Method": "RAG", "Atomic_Coverage_Score": 0.8, "Support_Rate": 0.8, "Atomic_Precision": 0.9, "Hallucination_Rate": 0.1, "Contradiction_Rate": 0.0, "Not_In_Scope_Rate": 0.2, "Likert_Score": 3, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q2", "Model": "m1", "Method": "RAG", "Atomic_Coverage_Score": 0.9, "Support_Rate": 0.9, "Atomic_Precision": 0.95, "Hallucination_Rate": 0.05, "Contradiction_Rate": 0.0, "Not_In_Scope_Rate": 0.1, "Likert_Score": 4, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
            {"ID": "q3", "Model": "m1", "Method": "RAG", "Atomic_Coverage_Score": 0.6, "Support_Rate": 0.6, "Atomic_Precision": 0.7, "Hallucination_Rate": 0.3, "Contradiction_Rate": 0.1, "Not_In_Scope_Rate": 0.3, "Likert_Score": 3, "Trap_Fallibility": 0.0, "Cost_USD": 0.0, "Dauer_sek": 1.0, "Is_Trap_Question": 0},
        ]
    )

    figure_files = generate_run_figures(evaluated_df, question_file, outdir)

    assert any("figure_1_model_category_score" in path for path in figure_files)
    assert any("figure_5_tasktype_delta_boxplots_atomic_accuracy" in path for path in figure_files)
    assert any("figure_6_tasktype_delta_heatmap_atomic_accuracy" in path for path in figure_files)
    assert any("figure_7_shapiro_diagnostics_atomic_accuracy" in path for path in figure_files)
    assert any("figure_8_atomic_precision_model_method" in path for path in figure_files)
    assert any("figure_9_hallucination_rate_model_method" in path for path in figure_files)
    assert any("figure_10_coverage_precision_scatter" in path for path in figure_files)
    assert any("figure_11_risk_composition" in path for path in figure_files)
    assert any("figure_12_tasktype_delta_boxplots_atomic_precision" in path for path in figure_files)
    assert any("figure_13_tasktype_delta_boxplots_hallucination_rate" in path for path in figure_files)
    assert (outdir / "summary_tasktype_deltas.csv").exists()
    assert (outdir / "summary_tasktype_deltas_atomic_precision.csv").exists()
    assert (outdir / "summary_tasktype_deltas_hallucination_rate.csv").exists()
    assert (outdir / "summary_precision_risk_metrics.csv").exists()
    assert (outdir / "figure_7_shapiro_diagnostics_atomic_accuracy_values.csv").exists()


def test_run_variance_summary_reports_metric_std_by_run():
    df = pd.DataFrame(
        [
            {"ID": "q1", "Model": "m1", "Method": "Zero-Shot", "Run_ID": 1, "Atomic_Coverage_Score": 0.5, "Atomic_Precision": 0.4, "Hallucination_Rate": 0.6, "Is_Trap_Question": 0},
            {"ID": "q1", "Model": "m1", "Method": "Zero-Shot", "Run_ID": 2, "Atomic_Coverage_Score": 0.7, "Atomic_Precision": 0.8, "Hallucination_Rate": 0.2, "Is_Trap_Question": 0},
            {"ID": "q2", "Model": "m1", "Method": "Zero-Shot", "Run_ID": 1, "Atomic_Coverage_Score": 0.3, "Atomic_Precision": 0.5, "Hallucination_Rate": 0.5, "Is_Trap_Question": 1},
            {"ID": "q2", "Model": "m1", "Method": "Zero-Shot", "Run_ID": 2, "Atomic_Coverage_Score": 0.9, "Atomic_Precision": 0.9, "Hallucination_Rate": 0.1, "Is_Trap_Question": 1},
        ]
    )

    summary = _build_run_variance_summary(df)
    precision_row = summary[summary["metric"] == "atomic_precision"].iloc[0]
    hallucination_row = summary[summary["metric"] == "hallucination_rate"].iloc[0]

    assert precision_row["n_tasks"] == 1
    assert abs(float(precision_row["mean_run_std"]) - 0.2828) < 1e-3
    assert abs(float(hallucination_row["mean_run_std"]) - 0.2828) < 1e-3
