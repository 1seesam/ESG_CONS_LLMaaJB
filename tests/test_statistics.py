import math

import pandas as pd

from src.benchmarking.statistics import benjamini_hochberg, build_stats_tables


def _base_row(model, method, qid, run_id, score, confidence):
    coverage = score / 4.0
    return {
        "Model": model,
        "Method": method,
        "ID": qid,
        "Run_ID": run_id,
        "Likert_Score": float(score),
        "KI_Antwort": f"{method} answer {qid} {run_id}",
        "Cost_USD": 0.01,
        "Dauer_sek": 1.0,
        "Atomic_Coverage_Score": coverage,
        "Support_Rate": coverage,
        "Atomic_Precision": coverage,
        "Hallucination_Rate": 1.0 - coverage,
        "Contradiction_Rate": 0.0,
        "Not_In_Scope_Rate": 0.0,
        "Trap_Pass": float("nan"),
        "Trap_Hallucination": float("nan"),
        "Trap_Fallibility": float("nan"),
        "Is_Trap_Question": 0,
        "Confidence_Mean": confidence,
        "Confidence_Found_Flag": 0 if pd.isna(confidence) else 1,
    }


def test_bh_correction_monotonic():
    q = benjamini_hochberg([0.01, 0.03, 0.2])
    assert q[0] <= q[1] <= q[2]


def test_build_stats_tables_has_expected_columns():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 1, 1),
            _base_row("m1", "RAG", "q1", 1, 3, 3),
            _base_row("m1", "Zero-Shot", "q2", 1, 2, 1),
            _base_row("m1", "RAG", "q2", 1, 4, 3),
        ]
    )
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    for col in [
        "p_value",
        "q_value",
        "effect_size",
        "ci95_low",
        "ci95_high",
        "test_name",
        "paired_n",
        "multiple_testing_scope",
        "confidence_mean",
        "overconfidence_score",
        "stability_check_note",
        "primary_metric_name",
    ]:
        assert col in stats.columns
    for col in ["Mean_Likert", "Trap_Fallibility_Rate", "Biz Value Index"]:
        assert col not in stats.columns


def test_build_stats_tables_handles_all_nan_scores():
    df = pd.DataFrame([_base_row("m1", "Zero-Shot", "q1", 1, 1, 1)])
    df["Likert_Score"] = float("nan")
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    assert stats.empty
    for col in ["Model", "Method", "Mean_Score", "p_value", "q_value"]:
        assert col in stats.columns


def test_wilcoxon_is_used_on_task_level_pairs():
    rows = []
    for qid, zs_score, rag_score in [("q1", 1, 4), ("q2", 1, 4), ("q3", 2, 4), ("q4", 1, 3)]:
        rows.append(_base_row("m1", "Zero-Shot", qid, 1, zs_score, 1))
        rows.append(_base_row("m1", "RAG", qid, 1, rag_score, 3))
    stats = build_stats_tables(pd.DataFrame(rows), baseline_method="Zero-Shot")
    rag_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "RAG")].iloc[0]
    assert rag_row["test_name"] == "wilcoxon_signed_rank"
    assert rag_row["analysis_unit"] == "task"
    assert rag_row["pairing_key"] == "Model+ID"
    assert rag_row["paired_n"] == 4
    assert not pd.isna(rag_row["p_value"])
    assert rag_row["effect_size"] > 0


def test_all_zero_differences_return_p_one_and_zero_effect():
    rows = []
    for qid, score in [("q1", 2), ("q2", 2), ("q3", 2)]:
        rows.append(_base_row("m1", "Zero-Shot", qid, 1, score, 2))
        rows.append(_base_row("m1", "RAG", qid, 1, score, 2))
    stats = build_stats_tables(pd.DataFrame(rows), baseline_method="Zero-Shot")
    rag_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "RAG")].iloc[0]
    assert rag_row["paired_n"] == 3
    assert rag_row["p_value"] == 1.0
    assert rag_row["effect_size"] == 0.0


def test_incomplete_pairs_yield_nan_p_value():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 1, 1),
            _base_row("m1", "RAG", "q1", 1, 4, 3),
            _base_row("m1", "Zero-Shot", "q2", 1, 2, 1),
            _base_row("m1", "RAG", "q3", 1, 4, 3),
        ]
    )
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    rag_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "RAG")].iloc[0]
    assert rag_row["paired_n"] == 1
    assert pd.isna(rag_row["p_value"])


def test_q_values_are_computed_per_model():
    rows = []
    for model in ["m1", "m2"]:
        for qid, zs, rag, adv in [("q1", 1, 4, 3), ("q2", 1, 4, 2), ("q3", 1, 3, 1), ("q4", 2, 4, 2)]:
            rows.append(_base_row(model, "Zero-Shot", qid, 1, zs, 1))
            rows.append(_base_row(model, "RAG", qid, 1, rag, 3))
            rows.append(_base_row(model, "Advanced-Prompt", qid, 1, adv, 2))
    stats = build_stats_tables(pd.DataFrame(rows), baseline_method="Zero-Shot")
    for model in ["m1", "m2"]:
        baseline_row = stats[(stats["Model"] == model) & (stats["Method"] == "Zero-Shot")].iloc[0]
        rag_row = stats[(stats["Model"] == model) & (stats["Method"] == "RAG")].iloc[0]
        adv_row = stats[(stats["Model"] == model) & (stats["Method"] == "Advanced-Prompt")].iloc[0]
        assert baseline_row["q_value"] == 1.0
        assert not pd.isna(rag_row["q_value"])
        assert not pd.isna(adv_row["q_value"])


def test_std_dev_and_run_level_std_are_task_based():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 1, 1),
            _base_row("m1", "Zero-Shot", "q1", 2, 1, 1),
            _base_row("m1", "Zero-Shot", "q2", 1, 2, 1),
            _base_row("m1", "Zero-Shot", "q2", 2, 4, 1),
            _base_row("m1", "RAG", "q1", 1, 3, 3),
            _base_row("m1", "RAG", "q1", 2, 3, 3),
            _base_row("m1", "RAG", "q2", 1, 4, 3),
            _base_row("m1", "RAG", "q2", 2, 4, 3),
        ]
    )
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    zs_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "Zero-Shot")].iloc[0]
    assert not pd.isna(zs_row["Std_Dev"])
    assert math.isclose(zs_row["Mean_Score"], 0.5, rel_tol=1e-6)
    assert math.isclose(zs_row["run_level_score_std"], math.sqrt(2) / 8, rel_tol=1e-6)
    assert math.isclose(zs_row["run_level_coverage_std"], zs_row["run_level_score_std"], rel_tol=1e-6)


def test_run_level_std_is_computed_for_precision_and_hallucination():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 2, 1),
            _base_row("m1", "Zero-Shot", "q1", 2, 2, 1),
            _base_row("m1", "Zero-Shot", "q2", 1, 2, 1),
            _base_row("m1", "Zero-Shot", "q2", 2, 2, 1),
        ]
    )
    df.loc[(df["ID"] == "q1") & (df["Run_ID"] == 1), "Atomic_Precision"] = 0.5
    df.loc[(df["ID"] == "q1") & (df["Run_ID"] == 2), "Atomic_Precision"] = 0.7
    df.loc[(df["ID"] == "q2") & (df["Run_ID"] == 1), "Atomic_Precision"] = 0.4
    df.loc[(df["ID"] == "q2") & (df["Run_ID"] == 2), "Atomic_Precision"] = 0.4
    df["Hallucination_Rate"] = 1.0 - df["Atomic_Precision"]

    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    zs_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "Zero-Shot")].iloc[0]

    assert math.isclose(zs_row["run_level_precision_std"], math.sqrt(0.02) / 2, rel_tol=1e-6)
    assert math.isclose(zs_row["run_level_hallucination_std"], math.sqrt(0.02) / 2, rel_tol=1e-6)


def test_confidence_and_overconfidence_are_aggregated():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 0, 3),
            _base_row("m1", "Zero-Shot", "q2", 1, 4, 3),
            _base_row("m1", "RAG", "q1", 1, 4, 3),
            _base_row("m1", "RAG", "q2", 1, 4, 3),
        ]
    )
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    zs_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "Zero-Shot")].iloc[0]
    assert math.isclose(zs_row["confidence_mean"], 3.0, rel_tol=1e-6)
    assert math.isclose(zs_row["overconfidence_score"], 0.0, rel_tol=1e-6)
    assert math.isclose(zs_row["confidence_accuracy_gap"], -0.5, rel_tol=1e-6)


def test_high_certainty_and_low_accuracy_increases_overconfidence():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 0, 0),
            _base_row("m1", "Zero-Shot", "q2", 1, 0, 0),
            _base_row("m1", "RAG", "q1", 1, 4, 3),
            _base_row("m1", "RAG", "q2", 1, 4, 3),
        ]
    )
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    zs_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "Zero-Shot")].iloc[0]
    assert math.isclose(zs_row["overconfidence_score"], 1.0, rel_tol=1e-6)
    assert math.isclose(zs_row["confidence_accuracy_gap"], 1.0, rel_tol=1e-6)


def test_trap_questions_do_not_affect_primary_score_but_do_affect_trap_metrics():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 4, 1),
            _base_row("m1", "Zero-Shot", "q2", 1, 0, 1),
            _base_row("m1", "RAG", "q1", 1, 0, 1),
            _base_row("m1", "RAG", "q2", 1, 0, 1),
        ]
    )
    df.loc[df["ID"] == "q2", "Is_Trap_Question"] = 1
    df.loc[df["ID"] == "q2", "Trap_Pass"] = 0.0
    df.loc[df["ID"] == "q2", "Trap_Hallucination"] = 1.0
    df.loc[df["ID"] == "q2", "Trap_Fallibility"] = 1.0
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    zs_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "Zero-Shot")].iloc[0]
    rag_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "RAG")].iloc[0]
    assert math.isclose(zs_row["Mean_Score"], 1.0, rel_tol=1e-6)
    assert math.isclose(rag_row["Mean_Score"], 0.0, rel_tol=1e-6)
    assert math.isclose(zs_row["Trap_Hallucination_Rate"], 1.0, rel_tol=1e-6)
    assert math.isclose(rag_row["Trap_Hallucination_Rate"], 1.0, rel_tol=1e-6)
    assert math.isclose(zs_row["Trap_Question_Share"], 0.5, rel_tol=1e-6)


def test_trap_questions_do_not_contribute_to_paired_n_or_wilcoxon():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 1, 1),
            _base_row("m1", "RAG", "q1", 1, 4, 1),
            _base_row("m1", "Zero-Shot", "q2", 1, 4, 1),
            _base_row("m1", "RAG", "q2", 1, 0, 1),
            _base_row("m1", "Zero-Shot", "q3", 1, 4, 1),
            _base_row("m1", "RAG", "q3", 1, 0, 1),
        ]
    )
    df.loc[df["ID"].isin(["q2", "q3"]), "Is_Trap_Question"] = 1
    df.loc[df["ID"].isin(["q2", "q3"]), "Trap_Pass"] = 0.0
    df.loc[df["ID"].isin(["q2", "q3"]), "Trap_Hallucination"] = 1.0
    df.loc[df["ID"].isin(["q2", "q3"]), "Trap_Fallibility"] = 1.0
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    rag_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "RAG")].iloc[0]
    assert rag_row["N"] == 1
    assert rag_row["paired_n"] == 1
    assert pd.isna(rag_row["p_value"])


def test_run_level_pair_coverage_ignores_trap_questions():
    df = pd.DataFrame(
        [
            _base_row("m1", "Zero-Shot", "q1", 1, 1, 1),
            _base_row("m1", "Zero-Shot", "q1", 2, 1, 1),
            _base_row("m1", "RAG", "q1", 1, 4, 1),
            _base_row("m1", "RAG", "q1", 2, 4, 1),
            _base_row("m1", "Zero-Shot", "q2", 1, 4, 1),
            _base_row("m1", "Zero-Shot", "q2", 2, 4, 1),
            _base_row("m1", "RAG", "q2", 1, 0, 1),
        ]
    )
    df.loc[df["ID"] == "q2", "Is_Trap_Question"] = 1
    df.loc[df["ID"] == "q2", "Trap_Pass"] = 0.0
    df.loc[df["ID"] == "q2", "Trap_Hallucination"] = 1.0
    df.loc[df["ID"] == "q2", "Trap_Fallibility"] = 1.0
    stats = build_stats_tables(df, baseline_method="Zero-Shot")
    rag_row = stats[(stats["Model"] == "m1") & (stats["Method"] == "RAG")].iloc[0]
    assert math.isclose(rag_row["run_level_pair_coverage"], 1.0, rel_tol=1e-6)
