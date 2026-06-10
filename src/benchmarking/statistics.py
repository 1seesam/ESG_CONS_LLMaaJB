from __future__ import annotations

import io
import os
import random
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - local environments may not have scipy
    scipy_stats = None

STATS_COLUMNS = [
    "Model",
    "Method",
    "Mean_Score",
    "Std_Dev",
    "Semantic_Consistency",
    "Critical_Fail",
    "Avg_Cost",
    "Avg_Time",
    "Atomic_Coverage_Score",
    "Support_Rate",
    "Atomic_Precision",
    "Hallucination_Rate",
    "Contradiction_Rate",
    "Not_In_Scope_Rate",
    "Trap_Pass_Rate",
    "Trap_Hallucination_Rate",
    "Trap_Question_Share",
    "N",
    "Total_Real_Cost",
    "p_value",
    "q_value",
    "effect_size",
    "ci95_low",
    "ci95_high",
    "test_name",
    "analysis_unit",
    "baseline_method",
    "comparison_method",
    "pairing_key",
    "paired_n",
    "multiple_testing_method",
    "multiple_testing_scope",
    "effect_size_name",
    "effect_size_interpretation",
    "confidence_mean",
    "confidence_std",
    "overconfidence_score",
    "confidence_accuracy_gap",
    "stability_check_name",
    "stability_check_note",
    "run_level_score_std",
    "run_level_coverage_std",
    "run_level_precision_std",
    "run_level_hallucination_std",
    "run_level_pair_coverage",
    "rq_mapping",
    "primary_metric_name",
]

THESIS_METHOD_ORDER = ["Zero-Shot", "Advanced-Prompt", "RAG"]
THESIS_METHOD_LABELS = {
    "Zero-Shot": "Zero-Shot",
    "Advanced-Prompt": "Advanced Prompting",
    "RAG": "RAG",
}
THESIS_METHOD_ALIASES = {
    "Zero-Shot": "Zero-Shot",
    "Advanced-Prompt": "Advanced-Prompt",
    "Advanced Prompt": "Advanced-Prompt",
    "Advanced Prompting": "Advanced-Prompt",
    "RAG": "RAG",
}
THESIS_MAIN_COLUMNS = [
    "model",
    "metric",
    "comparison",
    "n_pairs",
    "wilcoxon_statistic",
    "p_value_raw",
    "p_value_bh",
    "effect_size_r",
    "effect_size_label",
    "direction",
    "shapiro_p_value",
]
THESIS_TRAP_COLUMNS = [
    "model",
    "method",
    "n_trap_questions",
    "correct_refusal_rate",
    "hallucinated_answer_rate",
]
THESIS_ROBUSTNESS_COLUMNS = [
    "model",
    "metric",
    "comparison",
    "n_runs_with_pairs",
    "share_runs_same_direction",
    "share_runs_same_significance",
    "overall_direction_stable",
    "overall_significance_stable",
]
THESIS_OVERCONFIDENCE_COLUMNS = [
    "model",
    "method",
    "n_observations",
    "mean_confidence_score",
    "mean_atomic_coverage_score",
    "confidence_accuracy_correlation",
    "overconfident_error_rate",
]
THESIS_ROUNDING = {
    "wilcoxon_statistic": 4,
    "p_value_raw": 4,
    "p_value_bh": 4,
    "effect_size_r": 4,
    "shapiro_p_value": 4,
    "correct_refusal_rate": 4,
    "hallucinated_answer_rate": 4,
    "share_runs_same_direction": 4,
    "share_runs_same_significance": 4,
    "mean_confidence_score": 4,
    "mean_atomic_coverage_score": 4,
    "confidence_accuracy_correlation": 4,
    "overconfident_error_rate": 4,
}
THESIS_METRIC_SPECS = {
    "atomic_coverage_score": {
        "column": "Atomic_Coverage_Score",
        "higher_is_better": True,
    },
    "atomic_precision": {
        "column": "Atomic_Precision",
        "higher_is_better": True,
    },
    "hallucination_rate": {
        "column": "Hallucination_Rate",
        "higher_is_better": False,
    },
}
THESIS_COMPARISONS = [
    ("Zero-Shot", "Advanced-Prompt"),
    ("Zero-Shot", "RAG"),
    ("Advanced-Prompt", "RAG"),
]


def _empty_stats_table() -> pd.DataFrame:
    return pd.DataFrame(columns=STATS_COLUMNS)


def _empty_thesis_table(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _round_thesis_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column, decimals in THESIS_ROUNDING.items():
        if column in out.columns:
            out[column] = out[column].round(decimals)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05):
    vals = [float(v) for v in values if not pd.isna(v)]
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = random.Random(42)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(0, len(vals))] for _ in range(len(vals))]
        means.append(float(np.mean(sample)))
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return lo, hi


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if not pd.isna(p)]
    if not indexed:
        return [np.nan for _ in p_values]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adj = [np.nan for _ in p_values]
    prev = 1.0
    for rank, (idx, p) in enumerate(reversed(indexed), start=1):
        k = m - rank + 1
        q = min(prev, (m / k) * p)
        prev = q
        adj[idx] = q
    return adj


def _semantic_consistency(answers: Sequence[str]) -> float:
    valid = [str(a) for a in answers if "API ERROR" not in str(a).upper() and str(a).lower() != "nan"]
    if len(valid) < 2:
        return 0.0
    use_embedding_model = os.environ.get("THESIS_ENABLE_EMBEDDING_CONSISTENCY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_embedding_model:
        tokens = [set(v.lower().split()) for v in valid]
        if len(tokens) < 2:
            return 0.0
        sims = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                inter = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j]) or 1
                sims.append(inter / union)
        return float(np.mean(sims)) if sims else 0.0
    try:
        from sentence_transformers import SentenceTransformer, util
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        emb = model.encode(valid, convert_to_tensor=True)
        cos = util.cos_sim(emb, emb)
        n = len(valid)
        num = 0.0
        den = 0
        for i in range(n):
            for j in range(i + 1, n):
                num += float(cos[i][j].item())
                den += 1
        return num / den if den else 0.0
    except Exception:
        tokens = [set(v.lower().split()) for v in valid]
        if len(tokens) < 2:
            return 0.0
        sims = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                inter = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j]) or 1
                sims.append(inter / union)
        return float(np.mean(sims)) if sims else 0.0


def _effect_size_interpretation(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    abs_value = abs(float(value))
    if abs_value < 0.1:
        return "negligible"
    if abs_value < 0.3:
        return "small"
    if abs_value < 0.5:
        return "medium"
    return "large"


def _thesis_effect_size_label(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    abs_value = abs(float(value))
    if abs_value < 0.3:
        return "klein"
    if abs_value < 0.5:
        return "mittel"
    return "gross"


def _normalize_method_name(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return THESIS_METHOD_ALIASES.get(text, text)


def _export_method_name(value: object) -> str:
    method = _normalize_method_name(value)
    return THESIS_METHOD_LABELS.get(method, method)


def _comparison_label(method_a: str, method_b: str) -> str:
    return f"{_export_method_name(method_a)} vs {_export_method_name(method_b)}"


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(arr.size, dtype=float)
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _wilcoxon_two_sided_pvalue_exact(differences: Sequence[float]) -> float:
    diffs = np.asarray([float(v) for v in differences if not pd.isna(v)], dtype=float)
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 1.0
    ranks = _rankdata_average(np.abs(diffs))
    int_ranks = np.rint(ranks * 2).astype(int)
    observed = int(int_ranks[diffs > 0].sum())
    total = int(int_ranks.sum())
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for rank in int_ranks:
        next_counts = counts.copy()
        next_counts[rank:] += counts[:-rank]
        counts = next_counts
    total_assignments = 2 ** len(int_ranks)
    prob_le = counts[: observed + 1].sum() / total_assignments
    prob_ge = counts[observed:].sum() / total_assignments
    return float(min(1.0, 2.0 * min(prob_le, prob_ge)))


def _rank_biserial_from_differences(differences: Sequence[float]) -> float:
    diffs = np.asarray([float(v) for v in differences if not pd.isna(v)], dtype=float)
    if diffs.size == 0:
        return np.nan
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 0.0
    ranks = (
        scipy_stats.rankdata(np.abs(diffs), method="average")
        if scipy_stats is not None
        else _rankdata_average(np.abs(diffs))
    )
    pos = float(ranks[diffs > 0].sum())
    neg = float(ranks[diffs < 0].sum())
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _prepare_source(df: pd.DataFrame) -> pd.DataFrame:
    source = df.copy()
    defaults = {
        "Model": "unknown",
        "Method": "unknown",
        "ID": "unknown",
        "Run_ID": np.nan,
        "KI_Antwort": "",
        "Cost_USD": np.nan,
        "Dauer_sek": np.nan,
        "Atomic_Coverage_Score": np.nan,
        "Support_Rate": np.nan,
        "Atomic_Precision": np.nan,
        "Hallucination_Rate": np.nan,
        "Contradiction_Rate": np.nan,
        "Not_In_Scope_Rate": np.nan,
        "Trap_Pass": np.nan,
        "Trap_Hallucination": np.nan,
        "Trap_Fallibility": np.nan,
        "Is_Trap_Question": np.nan,
        "Confidence_Mean": np.nan,
        "Confidence_Found_Flag": 0,
    }
    for col, default in defaults.items():
        if col not in source.columns:
            source[col] = default
    source["Method"] = source["Method"].map(_normalize_method_name)
    source["Atomic_Coverage_Score"] = source["Atomic_Coverage_Score"].fillna(source["Support_Rate"])
    source["Support_Rate"] = source["Support_Rate"].fillna(source["Atomic_Coverage_Score"])
    if source["Hallucination_Rate"].isna().all() and not source["Atomic_Precision"].isna().all():
        source["Hallucination_Rate"] = 1.0 - source["Atomic_Precision"]
    source["Confidence_Found_Flag"] = source["Confidence_Found_Flag"].fillna(0).astype(int)
    # Prompt semantics: 0 = very certain/correct, 3 = highly uncertain / potentially wrong.
    # Convert the raw marker into a certainty score where higher means more confidence.
    source["Confidence_Norm"] = 1.0 - (source["Confidence_Mean"] / 3.0)
    source["Score_Norm"] = source["Support_Rate"]
    source["Overconfidence"] = source["Confidence_Norm"] * (1.0 - source["Score_Norm"])
    source.loc[source["Confidence_Mean"].isna(), "Overconfidence"] = np.nan
    source["Confidence_Accuracy_Gap"] = source["Confidence_Norm"] - source["Score_Norm"]
    source.loc[source["Confidence_Mean"].isna(), "Confidence_Accuracy_Gap"] = np.nan
    if "Likert_Score" not in source.columns:
        return source
    source = source.dropna(subset=["Likert_Score"]).copy()
    if source.empty:
        return source
    return source


def _aggregate_task_level(source: pd.DataFrame) -> pd.DataFrame:
    grouped = source.groupby(["Model", "Method", "ID"], dropna=False).apply(
        lambda x: pd.Series(
            {
                "Is_Trap_Question": float(x["Is_Trap_Question"].mean()),
                "Task_Score": x["Support_Rate"].mean(),
                "Task_Binary_Score": x["Support_Rate"].mean(),
                "Task_Run_Score_Std": x["Support_Rate"].std(),
                "Task_Run_Coverage_Std": x["Atomic_Coverage_Score"].std(),
                "Task_Run_Precision_Std": x["Atomic_Precision"].std(),
                "Task_Run_Hallucination_Std": x["Hallucination_Rate"].std(),
                "Semantic_Consistency": _semantic_consistency(x["KI_Antwort"].tolist()),
                "Critical_Fail": 1 if x["Support_Rate"].mean() < 0.5 else 0,
                "Avg_Cost": x["Cost_USD"].mean(),
                "Avg_Time": x["Dauer_sek"].mean(),
                "Atomic_Coverage_Score": x["Atomic_Coverage_Score"].mean(),
                "Support_Rate": x["Support_Rate"].mean(),
                "Atomic_Precision": x["Atomic_Precision"].mean(),
                "Hallucination_Rate": x["Hallucination_Rate"].mean(),
                "Contradiction_Rate": x["Contradiction_Rate"].mean(),
                "Not_In_Scope_Rate": x["Not_In_Scope_Rate"].mean(),
                "Trap_Pass_Rate": x["Trap_Pass"].mean(),
                "Trap_Hallucination_Rate": x["Trap_Hallucination"].mean(),
                "Trap_Question_Share": x["Is_Trap_Question"].mean(),
                "Task_Confidence_Mean": x["Confidence_Mean"].mean(),
                "Task_Confidence_Found_Rate": x["Confidence_Found_Flag"].mean(),
                "Task_Overconfidence_Score": x["Overconfidence"].mean(),
                "Task_Confidence_Accuracy_Gap": x["Confidence_Accuracy_Gap"].mean(),
            }
        ),
        include_groups=False,
    )
    return grouped.reset_index()


def _non_trap_task_df(task_df: pd.DataFrame) -> pd.DataFrame:
    if "Is_Trap_Question" not in task_df.columns:
        return task_df.copy()
    return task_df[task_df["Is_Trap_Question"].fillna(0) < 0.5].copy()


def _trap_task_df(task_df: pd.DataFrame) -> pd.DataFrame:
    if "Is_Trap_Question" not in task_df.columns:
        return task_df.iloc[0:0].copy()
    return task_df[task_df["Is_Trap_Question"].fillna(0) >= 0.5].copy()


def _paired_task_scores(
    task_df: pd.DataFrame,
    model: str,
    baseline_method: str,
    target_method: str,
) -> tuple[pd.Series, pd.Series]:
    base = (
        task_df[(task_df["Model"] == model) & (task_df["Method"] == baseline_method)][["ID", "Task_Binary_Score"]]
        .rename(columns={"Task_Binary_Score": "baseline_score"})
        .drop_duplicates(subset=["ID"])
    )
    target = (
        task_df[(task_df["Model"] == model) & (task_df["Method"] == target_method)][["ID", "Task_Binary_Score"]]
        .rename(columns={"Task_Binary_Score": "target_score"})
        .drop_duplicates(subset=["ID"])
    )
    merged = base.merge(target, on="ID", how="inner")
    return merged["baseline_score"], merged["target_score"]


def _perform_wilcoxon_test(
    task_df: pd.DataFrame,
    model: str,
    baseline_method: str,
    target_method: str,
) -> tuple[float, int, float]:
    baseline_scores, target_scores = _paired_task_scores(task_df, model, baseline_method, target_method)
    if baseline_scores.empty or target_scores.empty:
        return np.nan, 0, np.nan
    differences = target_scores.astype(float).values - baseline_scores.astype(float).values
    paired_n = int(len(differences))
    effect = _rank_biserial_from_differences(differences)
    if paired_n < 2:
        return np.nan, paired_n, effect
    if np.allclose(differences, 0.0):
        return 1.0, paired_n, 0.0
    try:
        if scipy_stats is not None:
            _, p_val = scipy_stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        else:
            p_val = _wilcoxon_two_sided_pvalue_exact(differences)
        return float(p_val), paired_n, effect
    except ValueError:
        return 1.0, paired_n, effect if not pd.isna(effect) else 0.0


def _compute_run_level_pair_coverage(
    source: pd.DataFrame,
    model: str,
    baseline_method: str,
    target_method: str,
) -> float:
    if "Run_ID" not in source.columns:
        return np.nan
    if target_method == baseline_method:
        return 1.0
    base = source[(source["Model"] == model) & (source["Method"] == baseline_method)]
    target = source[(source["Model"] == model) & (source["Method"] == target_method)]
    shared_ids = sorted(set(base["ID"]) & set(target["ID"]))
    if not shared_ids:
        return np.nan
    coverages = []
    for qid in shared_ids:
        base_runs = set(base[base["ID"] == qid]["Run_ID"].dropna().astype(str))
        target_runs = set(target[target["ID"] == qid]["Run_ID"].dropna().astype(str))
        denom = max(len(base_runs), len(target_runs))
        if denom == 0:
            continue
        coverages.append(len(base_runs & target_runs) / denom)
    if not coverages:
        return np.nan
    return float(np.mean(coverages))


def build_stats_tables(
    df: pd.DataFrame,
    baseline_method: str = "Zero-Shot",
    time_cost_factor: float = 0.002,
) -> pd.DataFrame:
    source = _prepare_source(df)
    if source.empty or "Likert_Score" not in source.columns:
        return _empty_stats_table()

    task_df = _aggregate_task_level(source)
    if task_df.empty:
        return _empty_stats_table()
    main_source = source[source["Is_Trap_Question"].fillna(0) < 0.5].copy()
    main_task_df = _non_trap_task_df(task_df)
    trap_task_df = _trap_task_df(task_df)
    if main_task_df.empty:
        return _empty_stats_table()

    final_stats = (
        main_task_df.groupby(["Model", "Method"], dropna=False)
        .agg(
            Mean_Score=("Task_Binary_Score", "mean"),
            Std_Dev=("Task_Binary_Score", "std"),
            Semantic_Consistency=("Semantic_Consistency", "mean"),
            Critical_Fail=("Critical_Fail", "mean"),
            Avg_Cost=("Avg_Cost", "mean"),
            Avg_Time=("Avg_Time", "mean"),
            Atomic_Coverage_Score=("Atomic_Coverage_Score", "mean"),
            Support_Rate=("Support_Rate", "mean"),
            Atomic_Precision=("Atomic_Precision", "mean"),
            Hallucination_Rate=("Hallucination_Rate", "mean"),
            Contradiction_Rate=("Contradiction_Rate", "mean"),
            Not_In_Scope_Rate=("Not_In_Scope_Rate", "mean"),
            N=("ID", "nunique"),
            confidence_mean=("Task_Confidence_Mean", "mean"),
            confidence_std=("Task_Confidence_Mean", "std"),
            overconfidence_score=("Task_Overconfidence_Score", "mean"),
            confidence_accuracy_gap=("Task_Confidence_Accuracy_Gap", "mean"),
            run_level_score_std=("Task_Run_Score_Std", "mean"),
            run_level_coverage_std=("Task_Run_Coverage_Std", "mean"),
            run_level_precision_std=("Task_Run_Precision_Std", "mean"),
            run_level_hallucination_std=("Task_Run_Hallucination_Std", "mean"),
        )
        .reset_index()
    )
    if final_stats.empty:
        return _empty_stats_table()

    trap_agg = (
        trap_task_df.groupby(["Model", "Method"], dropna=False)
        .agg(
            Trap_Pass_Rate=("Trap_Pass_Rate", "mean"),
            Trap_Hallucination_Rate=("Trap_Hallucination_Rate", "mean"),
            Trap_Question_Share=("ID", "nunique"),
        )
        .reset_index()
    )
    if not trap_agg.empty:
        final_stats = final_stats.merge(trap_agg, on=["Model", "Method"], how="left")
    else:
        final_stats["Trap_Pass_Rate"] = np.nan
        final_stats["Trap_Hallucination_Rate"] = np.nan
        final_stats["Trap_Question_Share"] = 0.0

    total_question_counts = (
        task_df.groupby(["Model", "Method"], dropna=False)["ID"].nunique().rename("Total_Questions").reset_index()
    )
    final_stats = final_stats.merge(total_question_counts, on=["Model", "Method"], how="left")
    final_stats["Trap_Question_Share"] = final_stats.apply(
        lambda row: (row["Trap_Question_Share"] / row["Total_Questions"])
        if pd.notna(row["Total_Questions"]) and row["Total_Questions"] > 0 and pd.notna(row["Trap_Question_Share"])
        else 0.0,
        axis=1,
    )
    final_stats = final_stats.drop(columns=["Total_Questions"])

    final_stats["Total_Real_Cost"] = final_stats["Avg_Cost"] + (final_stats["Avg_Time"] * time_cost_factor)
    final_stats["ci95_low"] = final_stats.apply(
        lambda row: bootstrap_ci(
            main_task_df[(main_task_df["Model"] == row["Model"]) & (main_task_df["Method"] == row["Method"])][
                "Task_Score"
            ].tolist()
        )[0],
        axis=1,
    )
    final_stats["ci95_high"] = final_stats.apply(
        lambda row: bootstrap_ci(
            main_task_df[(main_task_df["Model"] == row["Model"]) & (main_task_df["Method"] == row["Method"])][
                "Task_Score"
            ].tolist()
        )[1],
        axis=1,
    )
    final_stats["test_name"] = "wilcoxon_signed_rank"
    final_stats["analysis_unit"] = "task"
    final_stats["baseline_method"] = baseline_method
    final_stats["comparison_method"] = final_stats["Method"]
    final_stats["pairing_key"] = "Model+ID"
    final_stats["paired_n"] = 0
    final_stats["multiple_testing_method"] = "benjamini_hochberg"
    final_stats["multiple_testing_scope"] = "per_model"
    final_stats["effect_size_name"] = "rank_biserial_correlation"
    final_stats["effect_size_interpretation"] = "unknown"
    final_stats["stability_check_name"] = "run_level descriptive stability check"
    final_stats["stability_check_note"] = "descriptive stability check, not primary hypothesis testing"
    final_stats["run_level_pair_coverage"] = np.nan
    final_stats["rq_mapping"] = "RQ1=Atomic_Coverage_Score+Wilcoxon; RQ2=Precision+Hallucination+Overconfidence; RQ3=Cost/Time descriptive"
    final_stats["primary_metric_name"] = "atomic_coverage_score"
    final_stats["p_value"] = np.nan
    final_stats["q_value"] = np.nan
    final_stats["effect_size"] = np.nan
    final_stats["run_level_coverage_std"] = final_stats["run_level_coverage_std"].fillna(
        final_stats["run_level_score_std"]
    )
    final_stats["run_level_score_std"] = final_stats["run_level_score_std"].fillna(
        final_stats["run_level_coverage_std"]
    )

    for idx, row in final_stats.iterrows():
        model = row["Model"]
        method = row["Method"]
        final_stats.at[idx, "run_level_pair_coverage"] = _compute_run_level_pair_coverage(
            main_source,
            model=model,
            baseline_method=baseline_method,
            target_method=method,
        )
        if method == baseline_method:
            final_stats.at[idx, "test_name"] = "baseline_reference"
            final_stats.at[idx, "paired_n"] = int(row["N"])
            final_stats.at[idx, "p_value"] = 1.0
            final_stats.at[idx, "q_value"] = 1.0
            final_stats.at[idx, "effect_size"] = 0.0
            final_stats.at[idx, "effect_size_interpretation"] = _effect_size_interpretation(0.0)
            continue
        p_value, paired_n, effect_size = _perform_wilcoxon_test(
            main_task_df,
            model=model,
            baseline_method=baseline_method,
            target_method=method,
        )
        final_stats.at[idx, "paired_n"] = paired_n
        final_stats.at[idx, "p_value"] = p_value
        final_stats.at[idx, "effect_size"] = effect_size
        final_stats.at[idx, "effect_size_interpretation"] = _effect_size_interpretation(effect_size)

    for model in final_stats["Model"].dropna().unique():
        model_mask = final_stats["Model"] == model
        model_rows = final_stats[model_mask]
        non_baseline = model_rows["Method"] != baseline_method
        idxs = model_rows[non_baseline].index.tolist()
        p_values = [final_stats.at[idx, "p_value"] for idx in idxs]
        q_values = benjamini_hochberg(p_values)
        for idx, q_value in zip(idxs, q_values):
            final_stats.at[idx, "q_value"] = q_value
        baseline_idxs = model_rows[~non_baseline].index.tolist()
        for idx in baseline_idxs:
            final_stats.at[idx, "q_value"] = 1.0

    for column in STATS_COLUMNS:
        if column not in final_stats.columns:
            final_stats[column] = np.nan
    return final_stats[STATS_COLUMNS]


def _aggregate_thesis_task_level(source: pd.DataFrame) -> pd.DataFrame:
    grouped = source.groupby(["Model", "Method", "ID"], dropna=False).agg(
        Is_Trap_Question=("Is_Trap_Question", "mean"),
        Atomic_Coverage_Score=("Atomic_Coverage_Score", "mean"),
        Support_Rate=("Support_Rate", "mean"),
        Atomic_Precision=("Atomic_Precision", "mean"),
        Hallucination_Rate=("Hallucination_Rate", "mean"),
        Trap_Pass=("Trap_Pass", "mean"),
        Trap_Hallucination=("Trap_Hallucination", "mean"),
        Confidence_Norm=("Confidence_Norm", "mean"),
        Overconfidence=("Overconfidence", "mean"),
        Confidence_Mean=("Confidence_Mean", "mean"),
    )
    return grouped.reset_index()


def _aggregate_thesis_run_level(source: pd.DataFrame) -> pd.DataFrame:
    if "Run_ID" not in source.columns:
        return pd.DataFrame(
            columns=[
                "Model",
                "Method",
                "Run_ID",
                "ID",
                "Is_Trap_Question",
                "Atomic_Coverage_Score",
                "Support_Rate",
                "Atomic_Precision",
                "Hallucination_Rate",
            ]
        )
    grouped = source.groupby(["Model", "Method", "Run_ID", "ID"], dropna=False).agg(
        Is_Trap_Question=("Is_Trap_Question", "mean"),
        Atomic_Coverage_Score=("Atomic_Coverage_Score", "mean"),
        Support_Rate=("Support_Rate", "mean"),
        Atomic_Precision=("Atomic_Precision", "mean"),
        Hallucination_Rate=("Hallucination_Rate", "mean"),
    )
    return grouped.reset_index()


def _paired_metric_frame(
    frame: pd.DataFrame,
    model: str,
    method_a: str,
    method_b: str,
    metric_column: str,
) -> pd.DataFrame:
    left = (
        frame[(frame["Model"] == model) & (frame["Method"] == method_a)][["ID", metric_column]]
        .rename(columns={metric_column: "value_a"})
        .drop_duplicates(subset=["ID"])
    )
    right = (
        frame[(frame["Model"] == model) & (frame["Method"] == method_b)][["ID", metric_column]]
        .rename(columns={metric_column: "value_b"})
        .drop_duplicates(subset=["ID"])
    )
    return left.merge(right, on="ID", how="inner")


def _orient_differences(value_a: np.ndarray, value_b: np.ndarray, higher_is_better: bool) -> np.ndarray:
    raw = value_b.astype(float) - value_a.astype(float)
    if higher_is_better:
        return raw
    return -raw


def _wilcoxon_with_effect(differences: Sequence[float]) -> Dict[str, float]:
    diffs = np.asarray([float(v) for v in differences if not pd.isna(v)], dtype=float)
    if diffs.size == 0:
        return {
            "wilcoxon_statistic": np.nan,
            "p_value_raw": np.nan,
            "z_value": np.nan,
            "effect_size_r": np.nan,
            "n_nonzero": 0,
        }
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return {
            "wilcoxon_statistic": 0.0,
            "p_value_raw": 1.0,
            "z_value": 0.0,
            "effect_size_r": 0.0,
            "n_nonzero": 0,
        }
    ranks = (
        scipy_stats.rankdata(np.abs(nonzero), method="average")
        if scipy_stats is not None
        else _rankdata_average(np.abs(nonzero))
    )
    t_plus = float(ranks[nonzero > 0].sum())
    total_ranks = float(ranks.sum())
    t_minus = total_ranks - t_plus
    wilcoxon_statistic = float(min(t_plus, t_minus))
    if scipy_stats is not None:
        try:
            _, p_value = scipy_stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p_value = 1.0
    else:
        p_value = _wilcoxon_two_sided_pvalue_exact(nonzero)
    mean_t = total_ranks / 2.0
    var_t = float(np.sum(np.square(ranks)) / 4.0)
    z_value = 0.0 if var_t <= 0 else (t_plus - mean_t) / np.sqrt(var_t)
    effect_size_r = float(z_value / np.sqrt(nonzero.size))
    return {
        "wilcoxon_statistic": wilcoxon_statistic,
        "p_value_raw": float(p_value),
        "z_value": float(z_value),
        "effect_size_r": effect_size_r,
        "n_nonzero": int(nonzero.size),
    }


def _shapiro_pvalue(differences: Sequence[float]) -> float:
    if scipy_stats is None:
        return np.nan
    diffs = np.asarray([float(v) for v in differences if not pd.isna(v)], dtype=float)
    if diffs.size < 3:
        return np.nan
    if np.allclose(diffs, diffs[0]):
        return np.nan
    try:
        return float(scipy_stats.shapiro(diffs).pvalue)
    except ValueError:
        return np.nan


def _direction_label(differences: Sequence[float], method_a: str, method_b: str) -> str:
    diffs = np.asarray([float(v) for v in differences if not pd.isna(v)], dtype=float)
    if diffs.size == 0 or np.allclose(diffs, 0.0):
        return "tie"
    if float(np.mean(diffs)) > 0:
        return f"favours_{_export_method_name(method_b)}"
    return f"favours_{_export_method_name(method_a)}"


def build_thesis_main_tests(df: pd.DataFrame) -> pd.DataFrame:
    source = _prepare_source(df)
    if source.empty:
        return _empty_thesis_table(THESIS_MAIN_COLUMNS)
    task_df = _aggregate_thesis_task_level(source)
    main_task_df = _non_trap_task_df(task_df)
    models = sorted(source["Model"].dropna().astype(str).unique())
    rows: list[dict[str, object]] = []
    for model in models:
        for metric_name, metric_spec in THESIS_METRIC_SPECS.items():
            metric_column = metric_spec["column"]
            higher_is_better = bool(metric_spec["higher_is_better"])
            for method_a, method_b in THESIS_COMPARISONS:
                paired = _paired_metric_frame(main_task_df, model, method_a, method_b, metric_column)
                paired = paired.dropna(subset=["value_a", "value_b"])
                if paired.empty:
                    continue
                values_a = paired["value_a"].astype(float).to_numpy() if not paired.empty else np.asarray([], dtype=float)
                values_b = paired["value_b"].astype(float).to_numpy() if not paired.empty else np.asarray([], dtype=float)
                oriented_diffs = _orient_differences(values_a, values_b, higher_is_better)
                stats_payload = _wilcoxon_with_effect(oriented_diffs) if len(oriented_diffs) >= 2 else {
                    "wilcoxon_statistic": np.nan,
                    "p_value_raw": np.nan,
                    "z_value": np.nan,
                    "effect_size_r": np.nan,
                    "n_nonzero": 0,
                }
                rows.append(
                    {
                        "model": model,
                        "metric": metric_name,
                        "comparison": _comparison_label(method_a, method_b),
                        "n_pairs": int(len(oriented_diffs)),
                        "wilcoxon_statistic": stats_payload["wilcoxon_statistic"],
                        "p_value_raw": stats_payload["p_value_raw"],
                        "p_value_bh": np.nan,
                        "effect_size_r": stats_payload["effect_size_r"],
                        "effect_size_label": _thesis_effect_size_label(stats_payload["effect_size_r"]),
                        "direction": _direction_label(oriented_diffs, method_a, method_b),
                        "shapiro_p_value": _shapiro_pvalue(oriented_diffs),
                    }
                )
    main_df = pd.DataFrame(rows, columns=THESIS_MAIN_COLUMNS)
    if main_df.empty:
        return _empty_thesis_table(THESIS_MAIN_COLUMNS)
    for model in models:
        for metric_name in THESIS_METRIC_SPECS:
            mask = (main_df["model"] == model) & (main_df["metric"] == metric_name)
            p_values = main_df.loc[mask, "p_value_raw"].tolist()
            q_values = benjamini_hochberg(p_values)
            main_df.loc[mask, "p_value_bh"] = q_values
    return _round_thesis_table(main_df[THESIS_MAIN_COLUMNS])


def build_thesis_trap_summary(df: pd.DataFrame) -> pd.DataFrame:
    source = _prepare_source(df)
    if source.empty:
        return _empty_thesis_table(THESIS_TRAP_COLUMNS)
    task_df = _aggregate_thesis_task_level(source)
    models = sorted(source["Model"].dropna().astype(str).unique())
    rows: list[dict[str, object]] = []
    for model in models:
        available_methods = set(task_df[task_df["Model"] == model]["Method"].dropna().astype(str))
        for method in THESIS_METHOD_ORDER:
            if method not in available_methods:
                continue
            method_df = task_df[(task_df["Model"] == model) & (task_df["Method"] == method)]
            trap_df = method_df[method_df["Is_Trap_Question"].fillna(0) >= 0.5]
            rows.append(
                {
                    "model": model,
                    "method": _export_method_name(method),
                    "n_trap_questions": int(len(trap_df)),
                    "correct_refusal_rate": float(trap_df["Trap_Pass"].mean()) if not trap_df.empty else 0.0,
                    "hallucinated_answer_rate": (
                        float((1.0 - trap_df["Trap_Pass"]).mean()) if not trap_df.empty else 0.0
                    ),
                }
            )
    return _round_thesis_table(pd.DataFrame(rows, columns=THESIS_TRAP_COLUMNS))


def build_thesis_robustness_summary(df: pd.DataFrame, main_tests_df: pd.DataFrame) -> pd.DataFrame:
    source = _prepare_source(df)
    if source.empty or main_tests_df.empty or "Run_ID" not in source.columns:
        return _empty_thesis_table(THESIS_ROBUSTNESS_COLUMNS)
    run_df = _aggregate_thesis_run_level(source)
    run_df = _non_trap_task_df(run_df)
    rows: list[dict[str, object]] = []
    comparison_map = { _comparison_label(a, b): (a, b) for a, b in THESIS_COMPARISONS }
    for row in main_tests_df.to_dict(orient="records"):
        model = str(row["model"])
        metric = str(row["metric"])
        comparison = str(row["comparison"])
        methods = comparison_map.get(comparison)
        if methods is None:
            continue
        method_a, method_b = methods
        metric_spec = THESIS_METRIC_SPECS[metric]
        metric_column = metric_spec["column"]
        higher_is_better = bool(metric_spec["higher_is_better"])
        model_runs = sorted(
            run_df[run_df["Model"] == model]["Run_ID"].dropna().astype(str).unique()
        )
        overall_significant = bool(pd.notna(row["p_value_bh"]) and float(row["p_value_bh"]) < 0.05)
        run_matches_direction: list[bool] = []
        run_matches_significance: list[bool] = []
        for run_id in model_runs:
            run_slice = run_df[(run_df["Model"] == model) & (run_df["Run_ID"].astype(str) == run_id)]
        paired = _paired_metric_frame(run_slice, model, method_a, method_b, metric_column)
        paired = paired.dropna(subset=["value_a", "value_b"])
        if len(paired) < 2:
            continue
            oriented_diffs = _orient_differences(
                paired["value_a"].astype(float).to_numpy(),
                paired["value_b"].astype(float).to_numpy(),
                higher_is_better,
            )
            run_stats = _wilcoxon_with_effect(oriented_diffs)
            run_direction = _direction_label(oriented_diffs, method_a, method_b)
            run_significant = bool(pd.notna(run_stats["p_value_raw"]) and float(run_stats["p_value_raw"]) < 0.05)
            run_matches_direction.append(run_direction == row["direction"])
            run_matches_significance.append(run_significant == overall_significant)
        n_runs = len(run_matches_direction)
        share_direction = float(np.mean(run_matches_direction)) if run_matches_direction else np.nan
        share_significance = float(np.mean(run_matches_significance)) if run_matches_significance else np.nan
        rows.append(
            {
                "model": model,
                "metric": metric,
                "comparison": comparison,
                "n_runs_with_pairs": int(n_runs),
                "share_runs_same_direction": share_direction,
                "share_runs_same_significance": share_significance,
                "overall_direction_stable": bool(n_runs > 0 and share_direction == 1.0),
                "overall_significance_stable": bool(n_runs > 0 and share_significance == 1.0),
            }
        )
    return _round_thesis_table(pd.DataFrame(rows, columns=THESIS_ROBUSTNESS_COLUMNS))


def build_thesis_overconfidence_summary(df: pd.DataFrame) -> pd.DataFrame:
    source = _prepare_source(df)
    if source.empty:
        return _empty_thesis_table(THESIS_OVERCONFIDENCE_COLUMNS)
    non_trap = source[source["Is_Trap_Question"].fillna(0) < 0.5].copy()
    if non_trap.empty or non_trap["Confidence_Mean"].dropna().empty:
        return _empty_thesis_table(THESIS_OVERCONFIDENCE_COLUMNS)
    rows: list[dict[str, object]] = []
    grouped = non_trap.groupby(["Model", "Method"], dropna=False)
    for (model, method), frame in grouped:
        confidence = frame["Confidence_Norm"].astype(float)
        coverage = frame["Atomic_Coverage_Score"].astype(float)
        valid = pd.DataFrame({"confidence": confidence, "coverage": coverage}).dropna()
        corr = valid["confidence"].corr(valid["coverage"]) if len(valid) >= 2 else np.nan
        overconfident_errors = (
            (frame["Confidence_Norm"].astype(float) >= (2.0 / 3.0))
            & (frame["Atomic_Coverage_Score"].astype(float) < 0.5)
        )
        rows.append(
            {
                "model": str(model),
                "method": _export_method_name(method),
                "n_observations": int(len(frame)),
                "mean_confidence_score": float(confidence.mean()) if not confidence.dropna().empty else np.nan,
                "mean_atomic_coverage_score": float(coverage.mean()) if not coverage.dropna().empty else np.nan,
                "confidence_accuracy_correlation": float(corr) if pd.notna(corr) else np.nan,
                "overconfident_error_rate": float(overconfident_errors.mean()) if len(frame) else np.nan,
            }
        )
    return _round_thesis_table(pd.DataFrame(rows, columns=THESIS_OVERCONFIDENCE_COLUMNS))
