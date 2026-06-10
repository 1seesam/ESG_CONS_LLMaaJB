from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .statistics import _export_method_name, _thesis_effect_size_label, _wilcoxon_with_effect

try:  # pragma: no cover - optional dependency is declared for the thesis environment
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


TASK_TYPE_ORDER = [
    "Wissensbasiert",
    "Analytisch / Reasoning",
    "Rechen-/deterministisch",
    "Unklar/sonstige",
]
TASKTYPE_DELTA_COMPARISONS = [
    ("Zero-Shot", "Advanced-Prompt"),
    ("Zero-Shot", "RAG"),
]
SHAPIRO_DIAGNOSTIC_COMPARISONS = [
    ("Zero-Shot", "Advanced-Prompt"),
    ("Zero-Shot", "RAG"),
    ("Advanced-Prompt", "RAG"),
]
METHOD_COLOR_SEQUENCE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
COMPARISON_COLOR_SEQUENCE = ["#0072B2", "#D55E00", "#009E73"]
FALLBACK_COLOR = "#6E7781"
TEXT_COLOR = "#1F2933"
AXIS_COLOR = "#4B5563"
GRID_COLOR = "#D0D7DE"
BAR_EDGE_COLOR = "#FFFFFF"


def _short_model_label(model_name: object) -> str:
    label = str(model_name or "").strip()
    if "/" in label:
        label = label.split("/", 1)[1]
    if ":" in label:
        label = label.replace(":", "\n")
    if len(label) > 26:
        label = textwrap.fill(label, width=24, break_long_words=False, break_on_hyphens=True)
    return label


def _set_figure_text_style(fig: plt.Figure) -> None:
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily("DejaVu Sans")
        text.set_fontweight("normal")
        color = text.get_color()
        if isinstance(color, str) and color in {"black", "#000000"}:
            text.set_color(TEXT_COLOR)


def _method_color_map(methods: list[str]) -> dict[str, str]:
    return {
        method: color
        for method, color in zip(sorted(methods), METHOD_COLOR_SEQUENCE, strict=False)
    }


def _set_compact_xlabels(
    ax: plt.Axes,
    labels: list[str],
    *,
    rotation: int = 18,
    fontsize: int = 7,
) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([_short_model_label(label) for label in labels], rotation=rotation, ha="right", fontsize=fontsize)
    ax.tick_params(axis="x", pad=2, colors=TEXT_COLOR)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.85, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontweight("normal")


def _mark_zero_and_missing_values(
    ax: plt.Axes,
    x_positions: np.ndarray,
    values: np.ndarray,
    color: str,
) -> None:
    finite = np.isfinite(values)
    zero_mask = finite & np.isclose(values, 0.0)
    missing_mask = ~finite
    if zero_mask.any():
        ax.scatter(
            x_positions[zero_mask],
            np.full(int(zero_mask.sum()), 0.018),
            marker="_",
            s=170,
            linewidths=2.0,
            color=color,
            zorder=4,
        )
    if missing_mask.any():
        ax.scatter(
            x_positions[missing_mask],
            np.full(int(missing_mask.sum()), 0.035),
            marker="x",
            s=24,
            linewidths=1.1,
            color=FALLBACK_COLOR,
            zorder=4,
        )


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    _set_figure_text_style(fig)
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _tasktype_comparison_label(baseline_method: str, target_method: str) -> str:
    return f"{_export_method_name(target_method)} - {_export_method_name(baseline_method)}"


def _normalize_task_type(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unklar/sonstige"
    normalized = re.sub(r"\s+", " ", text.lower())
    if any(marker in normalized for marker in ["wissen", "wissensbasiert", "knowledge", "fakten"]):
        return "Wissensbasiert"
    if any(marker in normalized for marker in ["reasoning", "analyt", "analyse", "schluss", "logik"]):
        return "Analytisch / Reasoning"
    if any(marker in normalized for marker in ["kalkulation", "berechnung", "rechen", "determin"]):
        return "Rechen-/deterministisch"
    return "Unklar/sonstige"


def _load_question_meta(question_file: str | Path) -> pd.DataFrame:
    with Path(question_file).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return pd.DataFrame(columns=["ID", "Kategorie", "Schwierigkeitsgrad", "Aufgabentyp_Raw", "Aufgabentyp"])
    rows = []
    for row in payload:
        rows.append(
            {
                "ID": str(row.get("id", "")).strip(),
                "Kategorie": str(row.get("category", "")).strip(),
                "Schwierigkeitsgrad": str(row.get("difficulty", "")).strip(),
                "Aufgabentyp_Raw": str(row.get("task_type_raw", "")).strip(),
                "Aufgabentyp": _normalize_task_type(row.get("task_type_raw", "")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["ID", "Kategorie", "Schwierigkeitsgrad", "Aufgabentyp_Raw", "Aufgabentyp"])
    out["Kategorie"] = out["Kategorie"].replace({"": "Unbekannt"})
    out["Schwierigkeitsgrad"] = out["Schwierigkeitsgrad"].replace({"": "Unbekannt"})
    out["Aufgabentyp_Raw"] = out["Aufgabentyp_Raw"].replace({"": "Unklar/sonstige"})
    out["Aufgabentyp"] = out["Aufgabentyp"].replace({"": "Unklar/sonstige"})
    return out


def _prep_eval_df(evaluated_df: pd.DataFrame, question_file: str | Path) -> pd.DataFrame:
    df = evaluated_df.copy()
    required = [
        "ID",
        "Model",
        "Method",
        "Atomic_Coverage_Score",
        "Support_Rate",
        "Atomic_Precision",
        "Hallucination_Rate",
        "Contradiction_Rate",
        "Not_In_Scope_Rate",
        "Likert_Score",
        "Trap_Fallibility",
        "Cost_USD",
        "Dauer_sek",
        "Is_Trap_Question",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df["Atomic_Coverage_Score"] = df["Atomic_Coverage_Score"].fillna(df["Support_Rate"])
    df["Support_Rate"] = df["Support_Rate"].fillna(df["Atomic_Coverage_Score"])
    if df["Hallucination_Rate"].isna().all() and not df["Atomic_Precision"].isna().all():
        df["Hallucination_Rate"] = 1.0 - df["Atomic_Precision"]
    df["ID"] = df["ID"].astype(str)
    meta = _load_question_meta(question_file)
    df = df.merge(meta, on="ID", how="left")
    df["Kategorie"] = df["Kategorie"].fillna("Unbekannt")
    df["Schwierigkeitsgrad"] = df["Schwierigkeitsgrad"].fillna("Unbekannt")
    df["Aufgabentyp_Raw"] = df["Aufgabentyp_Raw"].fillna("Unklar/sonstige")
    df["Aufgabentyp"] = df["Aufgabentyp"].fillna("Unklar/sonstige")
    return df


TASKTYPE_DELTA_METRICS = {
    "atomic_coverage_score": {
        "column": "Atomic_Coverage_Score",
        "value_column": "task_atomic_accuracy",
        "delta_column": "delta_atomic_accuracy",
        "median_column": "median_delta_atomic_accuracy",
        "label": "Atomic Coverage Score",
        "filename_suffix": "atomic_accuracy",
    },
    "atomic_precision": {
        "column": "Atomic_Precision",
        "value_column": "task_atomic_precision",
        "delta_column": "delta_atomic_precision",
        "median_column": "median_delta_atomic_precision",
        "label": "Atomic Precision",
        "filename_suffix": "atomic_precision",
    },
    "hallucination_rate": {
        "column": "Hallucination_Rate",
        "value_column": "task_hallucination_rate",
        "delta_column": "delta_hallucination_rate",
        "median_column": "median_delta_hallucination_rate",
        "label": "Hallucination Rate",
        "filename_suffix": "hallucination_rate",
    },
}


def _metric_spec(metric_key: str) -> dict[str, str]:
    return TASKTYPE_DELTA_METRICS.get(metric_key, TASKTYPE_DELTA_METRICS["atomic_coverage_score"])


def _aggregate_non_trap_task_level_scores(df: pd.DataFrame, metric_key: str = "atomic_coverage_score") -> pd.DataFrame:
    spec = _metric_spec(metric_key)
    metric_col = spec["column"]
    value_col = spec["value_column"]
    non_trap = df[df["Is_Trap_Question"].fillna(0) < 0.5].copy()
    if non_trap.empty:
        return pd.DataFrame(columns=["Model", "Method", "ID", "Aufgabentyp", value_col])
    if metric_col not in non_trap.columns:
        if metric_key == "atomic_coverage_score" and "Support_Rate" in non_trap.columns:
            metric_col = "Support_Rate"
        else:
            return pd.DataFrame(columns=["Model", "Method", "ID", "Aufgabentyp", value_col])
    grouped = (
        non_trap.groupby(["Model", "Method", "ID", "Aufgabentyp"], dropna=False)[metric_col]
        .mean()
        .reset_index(name=value_col)
    )
    if grouped.empty:
        return pd.DataFrame(columns=["Model", "Method", "ID", "Aufgabentyp", value_col])
    return grouped


def _build_tasktype_delta_frame(df: pd.DataFrame, metric_key: str = "atomic_coverage_score") -> pd.DataFrame:
    spec = _metric_spec(metric_key)
    value_col = spec["value_column"]
    delta_col = spec["delta_column"]
    task_level = _aggregate_non_trap_task_level_scores(df, metric_key=metric_key)
    if task_level.empty:
        return pd.DataFrame(columns=["task_type", "comparison", delta_col, "model", "id"])

    rows: list[dict[str, object]] = []
    for baseline_method, target_method in TASKTYPE_DELTA_COMPARISONS:
        baseline = (
            task_level[task_level["Method"] == baseline_method][["Model", "ID", "Aufgabentyp", value_col]]
            .rename(columns={value_col: "baseline_score"})
            .drop_duplicates(subset=["Model", "ID", "Aufgabentyp"])
        )
        target = (
            task_level[task_level["Method"] == target_method][["Model", "ID", "Aufgabentyp", value_col]]
            .rename(columns={value_col: "target_score"})
            .drop_duplicates(subset=["Model", "ID", "Aufgabentyp"])
        )
        paired = baseline.merge(target, on=["Model", "ID", "Aufgabentyp"], how="inner").dropna(
            subset=["baseline_score", "target_score"]
        )
        if paired.empty:
            continue
        for _, pair in paired.iterrows():
            rows.append(
                {
                    "task_type": str(pair["Aufgabentyp"]),
                    "comparison": _tasktype_comparison_label(baseline_method, target_method),
                    delta_col: float(pair["target_score"]) - float(pair["baseline_score"]),
                    "model": str(pair["Model"]),
                    "id": str(pair["ID"]),
                }
            )
    deltas = pd.DataFrame(rows, columns=["task_type", "comparison", delta_col, "model", "id"])
    if deltas.empty:
        return deltas
    present_task_types = [task for task in TASK_TYPE_ORDER if task in set(deltas["task_type"].astype(str))]
    if present_task_types:
        deltas["task_type"] = pd.Categorical(deltas["task_type"], categories=present_task_types, ordered=True)
        deltas = deltas.sort_values(["comparison", "task_type", "model", "id"]).reset_index(drop=True)
    return deltas


def _comparison_label(method_a: str, method_b: str) -> str:
    return f"{_export_method_name(method_a)} vs {_export_method_name(method_b)}"


def _build_shapiro_diagnostic_frame(df: pd.DataFrame) -> pd.DataFrame:
    task_level = _aggregate_non_trap_task_level_scores(df)
    if task_level.empty:
        return pd.DataFrame(
            columns=[
                "comparison",
                "model",
                "id",
                "value_a",
                "value_b",
                "oriented_difference",
                "shapiro_p_value",
            ]
        )

    rows: list[dict[str, object]] = []
    for method_a, method_b in SHAPIRO_DIAGNOSTIC_COMPARISONS:
        left = (
            task_level[task_level["Method"] == method_a][["Model", "ID", "task_atomic_accuracy"]]
            .rename(columns={"task_atomic_accuracy": "value_a"})
            .drop_duplicates(subset=["Model", "ID"])
        )
        right = (
            task_level[task_level["Method"] == method_b][["Model", "ID", "task_atomic_accuracy"]]
            .rename(columns={"task_atomic_accuracy": "value_b"})
            .drop_duplicates(subset=["Model", "ID"])
        )
        paired = left.merge(right, on=["Model", "ID"], how="inner")
        if paired.empty:
            continue
        differences = paired["value_b"].astype(float).to_numpy() - paired["value_a"].astype(float).to_numpy()
        shapiro_p = np.nan
        if scipy_stats is not None and len(differences) >= 3 and not np.allclose(differences, differences[0]):
            try:
                shapiro_p = float(scipy_stats.shapiro(differences).pvalue)
            except ValueError:
                shapiro_p = np.nan
        comparison = _comparison_label(method_a, method_b)
        for (_, pair), diff in zip(paired.iterrows(), differences, strict=False):
            rows.append(
                {
                    "comparison": comparison,
                    "model": str(pair["Model"]),
                    "id": str(pair["ID"]),
                    "value_a": float(pair["value_a"]),
                    "value_b": float(pair["value_b"]),
                    "oriented_difference": float(diff),
                    "shapiro_p_value": shapiro_p,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "model",
            "id",
            "value_a",
            "value_b",
            "oriented_difference",
            "shapiro_p_value",
        ],
    )


def _round_tasktype_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in [
        "median_delta_atomic_accuracy",
        "median_delta_atomic_precision",
        "median_delta_hallucination_rate",
        "wilcoxon_statistic",
        "p_value_raw",
        "effect_size_r",
    ]:
        if column in out.columns:
            out[column] = out[column].round(4)
    return out


def _build_tasktype_delta_summary(df: pd.DataFrame, metric_key: str = "atomic_coverage_score") -> pd.DataFrame:
    spec = _metric_spec(metric_key)
    delta_col = spec["delta_column"]
    median_col = spec["median_column"]
    deltas = _build_tasktype_delta_frame(df, metric_key=metric_key)
    if deltas.empty:
        return pd.DataFrame(
            columns=[
                "task_type",
                "comparison",
                "n_pairs",
                median_col,
                "wilcoxon_statistic",
                "p_value_raw",
                "effect_size_r",
                "effect_size_label",
            ]
        )
    rows: list[dict[str, object]] = []
    for comparison in deltas["comparison"].dropna().astype(str).unique():
        comparison_df = deltas[deltas["comparison"] == comparison]
        for task_type in comparison_df["task_type"].dropna().astype(str).unique():
            task_df = comparison_df[comparison_df["task_type"].astype(str) == task_type]
            values = task_df[delta_col].astype(float).to_numpy()
            stats_payload = _wilcoxon_with_effect(values) if len(values) >= 2 else {
                "wilcoxon_statistic": np.nan,
                "p_value_raw": np.nan,
                "effect_size_r": np.nan,
            }
            rows.append(
                {
                    "task_type": task_type,
                    "comparison": comparison,
                    "n_pairs": int(len(values)),
                    median_col: float(np.median(values)) if len(values) else np.nan,
                    "wilcoxon_statistic": stats_payload["wilcoxon_statistic"],
                    "p_value_raw": stats_payload["p_value_raw"],
                    "effect_size_r": stats_payload["effect_size_r"],
                    "effect_size_label": _thesis_effect_size_label(stats_payload["effect_size_r"]),
                }
            )
    summary = pd.DataFrame(
        rows,
        columns=[
            "task_type",
            "comparison",
            "n_pairs",
            median_col,
            "wilcoxon_statistic",
            "p_value_raw",
            "effect_size_r",
            "effect_size_label",
        ],
    )
    if summary.empty:
        return summary
    present_task_types = [task for task in TASK_TYPE_ORDER if task in set(summary["task_type"].astype(str))]
    if present_task_types:
        summary["task_type"] = pd.Categorical(summary["task_type"], categories=present_task_types, ordered=True)
        summary = summary.sort_values(["comparison", "task_type"]).reset_index(drop=True)
    return _round_tasktype_summary(summary)


def _plot_model_by_category(df: pd.DataFrame, metric_col: str, title: str, outdir: Path, filename: str) -> None:
    g = (
        df.groupby(["Kategorie", "Model", "Method"], dropna=False)[metric_col]
        .mean()
        .reset_index()
        .sort_values(["Kategorie", "Model", "Method"])
    )
    if g.empty:
        return

    y_label_map = {
        "Support_Rate": "Mittlerer Atomic Coverage Score",
        "Trap_Fallibility": "Mittlere Anfälligkeit für Fangfragen",
    }
    y_label_map.update(
        {
            "Support_Rate": "Mittlerer Atomic Coverage Score",
            "Atomic_Coverage_Score": "Mittlerer Atomic Coverage Score",
            "Atomic_Precision": "Mittlere Atomic Precision",
            "Hallucination_Rate": "Mittlere Hallucination Rate",
        }
    )
    y_label = y_label_map.get(metric_col, metric_col.replace("_", " "))

    categories = list(g["Kategorie"].dropna().unique())
    top_n = 6
    categories = categories[:top_n]
    g = g[g["Kategorie"].isin(categories)].copy()
    categories = list(g["Kategorie"].unique())

    n = len(categories)
    cols = min(3, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), squeeze=False, sharey=True)
    method_colors = _method_color_map(list(g["Method"].dropna().unique()))

    for i, cat in enumerate(categories):
        ax = axes[i // cols][i % cols]
        dcat = g[g["Kategorie"] == cat]
        models = list(dcat["Model"].dropna().unique())
        methods = sorted(dcat["Method"].dropna().unique())
        x = np.arange(len(models))
        width = 0.78 / max(len(methods), 1)
        for j, method in enumerate(methods):
            d = dcat[dcat["Method"] == method].set_index("Model").reindex(models)
            y = d[metric_col].astype(float).values
            offset = (j - (len(methods) - 1) / 2) * width
            x_positions = x + offset
            color = method_colors.get(method, FALLBACK_COLOR)
            ax.bar(
                x_positions,
                y,
                width=width,
                color=color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.6,
                label=method,
            )
            _mark_zero_and_missing_values(ax, x_positions, y, color)
        ax.set_title(str(cat), fontsize=10, fontweight="normal", pad=8)
        _set_compact_xlabels(ax, models, rotation=24, fontsize=7)
        if i % cols == 0:
            ax.set_ylabel(y_label, labelpad=2)
        if metric_col in {"Support_Rate", "Atomic_Coverage_Score", "Atomic_Precision", "Hallucination_Rate", "Trap_Fallibility"}:
            ax.set_ylim(0, 1.05)
        _style_axis(ax)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(title, y=0.985, fontsize=13, fontweight="normal", color=TEXT_COLOR)
    fig.supxlabel("Modell", y=0.035, fontsize=10, color=TEXT_COLOR)
    fig.subplots_adjust(top=0.84, bottom=0.19, left=0.13, right=0.98, hspace=0.72, wspace=0.24)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Methode",
            loc="upper center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
            labelcolor=TEXT_COLOR,
        )
    _save(fig, outdir, filename)


def _plot_difficulty_aggregated(df: pd.DataFrame, outdir: Path) -> None:
    g = (
        df.groupby(["Schwierigkeitsgrad", "Method"], dropna=False)["Atomic_Coverage_Score"]
        .mean()
        .reset_index()
        .sort_values(["Schwierigkeitsgrad", "Method"])
    )
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    levels = sorted(g["Schwierigkeitsgrad"].dropna().unique(), key=lambda x: str(x))
    methods = sorted(g["Method"].dropna().unique())
    method_colors = _method_color_map(methods)
    x = np.arange(len(levels))
    for method in methods:
        d = g[g["Method"] == method].set_index("Schwierigkeitsgrad").reindex(levels)
        y = d["Atomic_Coverage_Score"].astype(float).values
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=method,
            color=method_colors.get(method, FALLBACK_COLOR),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=9, color=TEXT_COLOR)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Schwierigkeitsgrad")
    ax.set_ylabel("Mittlerer Atomic Coverage Score")
    ax.set_title("Atomic Coverage Score nach Schwierigkeitsgrad", fontweight="normal")
    _style_axis(ax)
    ax.legend(title="Methode", frameon=False, loc="best", labelcolor=TEXT_COLOR)
    _save(fig, outdir, "figure_3_difficulty_aggregated")


def _plot_costs_separate(df: pd.DataFrame, outdir: Path) -> None:
    g = (
        df.groupby(["Model", "Method"], dropna=False)["Cost_USD"]
        .mean()
        .reset_index()
        .sort_values(["Model", "Method"])
    )
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(g["Model"].dropna().unique())
    methods = sorted(g["Method"].dropna().unique())
    method_colors = _method_color_map(methods)
    x = np.arange(len(models))
    width = 0.78 / max(len(methods), 1)
    for j, method in enumerate(methods):
        d = g[g["Method"] == method].set_index("Model").reindex(models)
        y = d["Cost_USD"].astype(float).values
        offset = (j - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            y,
            width=width,
            label=method,
            color=method_colors.get(method, FALLBACK_COLOR),
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.6,
        )
    _set_compact_xlabels(ax, models, rotation=24, fontsize=8)
    ax.set_xlabel("Modell")
    ax.set_ylabel("Mittlere Kosten [USD]")
    ax.set_title("Kosten nach Modell und Methode", fontweight="normal")
    _style_axis(ax)
    ax.legend(title="Methode", frameon=False, labelcolor=TEXT_COLOR)
    _save(fig, outdir, "figure_4_costs")


def _plot_task_type_delta_boxplots(
    delta_df: pd.DataFrame,
    outdir: Path,
    metric_key: str = "atomic_coverage_score",
    filename: str | None = None,
) -> None:
    spec = _metric_spec(metric_key)
    delta_col = spec["delta_column"]
    label = spec["label"]
    if delta_df.empty or delta_col not in delta_df.columns:
        return
    comparisons = [
        _tasktype_comparison_label(baseline, target)
        for baseline, target in TASKTYPE_DELTA_COMPARISONS
        if _tasktype_comparison_label(baseline, target) in set(delta_df["comparison"].astype(str))
    ]
    task_types = [task for task in TASK_TYPE_ORDER if task in set(delta_df["task_type"].astype(str))]
    if not task_types:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(task_types))
    width = 0.34
    comparison_colors = {
        comparison: color
        for comparison, color in zip(comparisons, COMPARISON_COLOR_SEQUENCE, strict=False)
    }
    legend_handles: list[Patch] = []

    for idx, comparison in enumerate(comparisons):
        data = []
        for task_type in task_types:
            values = (
                delta_df[
                    (delta_df["comparison"].astype(str) == comparison)
                    & (delta_df["task_type"].astype(str) == task_type)
                ][delta_col]
                .astype(float)
                .dropna()
                .tolist()
            )
            data.append(values if values else [np.nan])
        positions = x + (idx - (len(comparisons) - 1) / 2) * width
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.78,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            medianprops={"color": TEXT_COLOR, "linewidth": 1.5},
            whiskerprops={"color": AXIS_COLOR, "linewidth": 1.0},
            capprops={"color": AXIS_COLOR, "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "edgecolor": AXIS_COLOR},
        )
        color = comparison_colors.get(comparison, FALLBACK_COLOR)
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
            patch.set_edgecolor(AXIS_COLOR)
        legend_handles.append(Patch(facecolor=color, edgecolor=AXIS_COLOR, alpha=0.82, label=comparison))
    ax.axhline(0.0, color=TEXT_COLOR, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(task_types, rotation=18, ha="right", fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel("Delta des Atomic Coverage Score gegenüber Zero-Shot")
    ax.set_xlabel("Aufgabentyp")
    ax.set_title("Verteilung der Delta-Werte des Atomic Coverage Score nach Aufgabentyp", fontweight="normal")
    ax.set_ylabel("Delta des Atomic Coverage Score gegenÃ¼ber Zero-Shot")
    ax.set_title("Verteilung der Delta-Werte des Atomic Coverage Score nach Aufgabentyp", fontweight="normal")
    ax.set_ylabel(f"Delta der {label} gegenueber Zero-Shot")
    title = f"Verteilung der Delta-Werte der {label} nach Aufgabentyp"
    if metric_key == "hallucination_rate":
        title += " (negative Werte = weniger Risiko)"
    ax.set_title(title, fontweight="normal")
    _style_axis(ax)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Vergleich", frameon=False, loc="best", labelcolor=TEXT_COLOR)
    _save(fig, outdir, filename or f"figure_5_tasktype_delta_boxplots_{spec['filename_suffix']}")


def _plot_task_type_delta_heatmap(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        return
    task_types = [task for task in TASK_TYPE_ORDER if task in set(summary["task_type"].astype(str))]
    if not task_types:
        return
    comparisons = [
        _tasktype_comparison_label(baseline, target)
        for baseline, target in TASKTYPE_DELTA_COMPARISONS
        if _tasktype_comparison_label(baseline, target) in set(summary["comparison"].astype(str))
    ]
    if not comparisons:
        return
    matrix = (
        summary.pivot(index="comparison", columns="task_type", values="median_delta_atomic_accuracy")
        .reindex(index=comparisons, columns=task_types)
    )
    values = matrix.astype(float).values
    finite_values = values[np.isfinite(values)]
    max_abs = max(float(np.max(np.abs(finite_values))), 0.05) if finite_values.size else 0.05
    fig, ax = plt.subplots(figsize=(2.4 * len(task_types) + 2.2, 3.4))
    heat = ax.imshow(values, aspect="auto", vmin=-max_abs, vmax=max_abs, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(task_types)))
    ax.set_xticklabels(task_types, rotation=18, ha="right", fontsize=9, color=TEXT_COLOR)
    ax.set_yticks(np.arange(len(comparisons)))
    ax.set_yticklabels(comparisons, fontsize=9, color=TEXT_COLOR)
    for row_idx, comparison in enumerate(comparisons):
        for col_idx, task_type in enumerate(task_types):
            value = matrix.loc[comparison, task_type]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{float(value):.2f}", ha="center", va="center", fontsize=8, color=TEXT_COLOR)
    cbar = fig.colorbar(heat, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Medianes Delta des Atomic Coverage Score")
    cbar.set_label("Medianes Delta des Atomic Coverage Score")
    ax.set_xlabel("Aufgabentyp")
    ax.set_ylabel("Vergleich")
    ax.set_title("Medianes Delta des Atomic Coverage Score nach Aufgabentyp", fontweight="normal")
    _style_axis(ax)
    fig.subplots_adjust(top=0.86, bottom=0.22)
    _save(fig, outdir, "figure_6_tasktype_delta_heatmap_atomic_accuracy")


def _plot_shapiro_diagnostics(diagnostic_df: pd.DataFrame, outdir: Path) -> None:
    if diagnostic_df.empty:
        return
    comparisons = [
        _comparison_label(method_a, method_b)
        for method_a, method_b in SHAPIRO_DIAGNOSTIC_COMPARISONS
        if _comparison_label(method_a, method_b) in set(diagnostic_df["comparison"].astype(str))
    ]
    if not comparisons:
        return

    fig, axes = plt.subplots(
        len(comparisons),
        2,
        figsize=(11.5, max(3.1 * len(comparisons), 5.2)),
        squeeze=False,
    )
    comparison_colors = {
        comparison: color
        for comparison, color in zip(comparisons, COMPARISON_COLOR_SEQUENCE, strict=False)
    }

    for row_idx, comparison in enumerate(comparisons):
        values = (
            diagnostic_df[diagnostic_df["comparison"].astype(str) == comparison]["oriented_difference"]
            .astype(float)
            .dropna()
            .to_numpy()
        )
        if values.size == 0:
            continue
        comparison_df = diagnostic_df[diagnostic_df["comparison"].astype(str) == comparison]
        shapiro_values = comparison_df["shapiro_p_value"].dropna()
        shapiro_text = f"{float(shapiro_values.iloc[0]):.4f}" if not shapiro_values.empty else "n/a"
        n_unique_ids = int(comparison_df["id"].astype(str).nunique())
        n_total_values = int(len(values))
        color = comparison_colors.get(comparison, FALLBACK_COLOR)
        title = f"{comparison} | n={n_unique_ids} je Modell | Shapiro-p={shapiro_text}"

        hist_ax = axes[row_idx][0]
        lower = min(-1.0, float(np.min(values)) - 0.05)
        upper = max(1.0, float(np.max(values)) + 0.05)
        bins = np.linspace(lower, upper, 9)
        hist_ax.hist(values, bins=bins, color=color, alpha=0.82, edgecolor=BAR_EDGE_COLOR, linewidth=0.7)
        hist_ax.axvline(0.0, color=TEXT_COLOR, linewidth=1.1, linestyle="--")
        hist_ax.set_title(title, fontsize=11, fontweight="normal", pad=8)
        hist_ax.text(
            0.99,
            0.98,
            f"{n_total_values} Beobachtungen gesamt",
            transform=hist_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=AXIS_COLOR,
        )
        hist_ax.set_xlabel("Gepaarte Differenz des Atomic Coverage Score")
        hist_ax.set_xlabel("Gepaarte Differenz des Atomic Coverage Score")
        hist_ax.set_ylabel("Anzahl")
        _style_axis(hist_ax)

        qq_ax = axes[row_idx][1]
        if scipy_stats is not None and len(values) >= 2:
            osm, osr = scipy_stats.probplot(values, dist="norm", fit=False)
            slope, intercept, _ = scipy_stats.probplot(values, dist="norm", fit=True)[1]
            qq_ax.scatter(osm, osr, color=color, s=42, alpha=0.92)
            x = np.asarray([min(osm), max(osm)], dtype=float)
            qq_ax.plot(x, slope * x + intercept, color=TEXT_COLOR, linewidth=1.2)
        else:
            qq_ax.scatter(np.arange(len(values)), np.sort(values), color=color, s=42, alpha=0.92)
        qq_ax.set_title("Q-Q-Plot der Differenzen", fontsize=11, fontweight="normal", pad=8)
        qq_ax.set_xlabel("Theoretische Normalquantile")
        qq_ax.set_ylabel("Beobachtete Differenzen")
        _style_axis(qq_ax)

    fig.suptitle(
        "Shapiro-Wilk-Diagnostik der gepaarten Atomic-Accuracy-Differenzen",
        y=0.985,
        fontsize=13,
        fontweight="normal",
        color=TEXT_COLOR,
    )
    fig.suptitle(
        "Shapiro-Wilk-Diagnostik der gepaarten Atomic-Coverage-Differenzen",
        y=0.985,
        fontsize=13,
        fontweight="normal",
        color=TEXT_COLOR,
    )
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.08, right=0.98, hspace=0.62, wspace=0.24)
    _save(fig, outdir, "figure_7_shapiro_diagnostics_atomic_accuracy")


def _build_precision_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Atomic_Coverage_Score",
        "Atomic_Precision",
        "Hallucination_Rate",
        "Contradiction_Rate",
        "Not_In_Scope_Rate",
    ]
    non_trap = df[df["Is_Trap_Question"].fillna(0) < 0.5].copy()
    if non_trap.empty:
        return pd.DataFrame(columns=["Model", "Method", *metric_cols, "N"])
    grouped = non_trap.groupby(["Model", "Method"], dropna=False).agg(
        Atomic_Coverage_Score=("Atomic_Coverage_Score", "mean"),
        Atomic_Precision=("Atomic_Precision", "mean"),
        Hallucination_Rate=("Hallucination_Rate", "mean"),
        Contradiction_Rate=("Contradiction_Rate", "mean"),
        Not_In_Scope_Rate=("Not_In_Scope_Rate", "mean"),
        N=("ID", "nunique"),
    )
    return grouped.reset_index()


def _plot_model_method_bar(summary: pd.DataFrame, metric_col: str, ylabel: str, title: str, outdir: Path, filename: str) -> None:
    g = summary.dropna(subset=[metric_col]).copy()
    if g.empty:
        return
    g = g.sort_values(["Model", "Method"])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    models = list(g["Model"].dropna().unique())
    methods = sorted(g["Method"].dropna().unique())
    method_colors = _method_color_map(methods)
    x = np.arange(len(models))
    width = 0.78 / max(len(methods), 1)
    for idx, method in enumerate(methods):
        d = g[g["Method"] == method].set_index("Model").reindex(models)
        y = d[metric_col].astype(float).values
        offset = (idx - (len(methods) - 1) / 2) * width
        x_positions = x + offset
        color = method_colors.get(method, FALLBACK_COLOR)
        ax.bar(x_positions, y, width=width, label=method, color=color, edgecolor=BAR_EDGE_COLOR, linewidth=0.6)
        _mark_zero_and_missing_values(ax, x_positions, y, color)
    _set_compact_xlabels(ax, models, rotation=24, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Modell")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="normal")
    _style_axis(ax)
    ax.legend(title="Methode", frameon=False, labelcolor=TEXT_COLOR)
    _save(fig, outdir, filename)


def _plot_coverage_precision_scatter(summary: pd.DataFrame, outdir: Path) -> None:
    g = summary.dropna(subset=["Atomic_Coverage_Score", "Atomic_Precision"]).copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    methods = sorted(g["Method"].dropna().unique())
    method_colors = _method_color_map(methods)
    for method in methods:
        d = g[g["Method"] == method]
        ax.scatter(
            d["Atomic_Coverage_Score"].astype(float),
            d["Atomic_Precision"].astype(float),
            s=72,
            alpha=0.88,
            label=method,
            color=method_colors.get(method, FALLBACK_COLOR),
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.7,
        )
        for _, row in d.iterrows():
            ax.annotate(
                _short_model_label(row["Model"]).split("\n")[0],
                (float(row["Atomic_Coverage_Score"]), float(row["Atomic_Precision"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=TEXT_COLOR,
            )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Atomic Coverage Score")
    ax.set_ylabel("Atomic Precision")
    ax.set_title("Coverage vs. Precision nach Modell und Methode", fontweight="normal")
    _style_axis(ax)
    ax.legend(title="Methode", frameon=False, labelcolor=TEXT_COLOR)
    _save(fig, outdir, "figure_10_coverage_precision_scatter")


def _plot_risk_composition(summary: pd.DataFrame, outdir: Path) -> None:
    metric_cols = ["Contradiction_Rate", "Not_In_Scope_Rate", "Hallucination_Rate"]
    g = summary.dropna(subset=metric_cols, how="all").copy()
    if g.empty:
        return
    g["label"] = g["Model"].map(_short_model_label) + "\n" + g["Method"].astype(str)
    x = np.arange(len(g))
    colors = {
        "Contradiction_Rate": "#D55E00",
        "Not_In_Scope_Rate": "#0072B2",
        "Hallucination_Rate": "#CC79A7",
    }
    labels = {
        "Contradiction_Rate": "Contradiction Rate",
        "Not_In_Scope_Rate": "Not-in-Scope Rate",
        "Hallucination_Rate": "Hallucination Rate",
    }
    fig, ax = plt.subplots(figsize=(max(9.0, 0.8 * len(g)), 5.4))
    bottom = np.zeros(len(g), dtype=float)
    for col in metric_cols:
        y = g[col].fillna(0).astype(float).to_numpy()
        ax.bar(x, y, bottom=bottom, color=colors[col], edgecolor=BAR_EDGE_COLOR, linewidth=0.5, label=labels[col])
        bottom += y
    ax.set_xticks(x)
    ax.set_xticklabels(g["label"].tolist(), rotation=24, ha="right", fontsize=7, color=TEXT_COLOR)
    ax.set_ylim(0, max(1.05, float(np.nanmax(bottom)) + 0.05))
    ax.set_xlabel("Modell / Methode")
    ax.set_ylabel("Mittlere Rate")
    ax.set_title("Risk Composition aus Gold-Luecken und Modell-Claim-Risiko", fontweight="normal")
    _style_axis(ax)
    ax.legend(frameon=False, loc="best", labelcolor=TEXT_COLOR)
    _save(fig, outdir, "figure_11_risk_composition")


def _build_run_variance_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Run_ID" not in df.columns:
        return pd.DataFrame(columns=["Model", "Method", "metric", "mean_run_std", "median_run_std", "n_tasks"])
    non_trap = df[df["Is_Trap_Question"].fillna(0) < 0.5].copy()
    if non_trap.empty:
        return pd.DataFrame(columns=["Model", "Method", "metric", "mean_run_std", "median_run_std", "n_tasks"])
    metric_map = {
        "atomic_coverage_score": "Atomic_Coverage_Score",
        "atomic_precision": "Atomic_Precision",
        "hallucination_rate": "Hallucination_Rate",
    }
    rows: list[dict[str, object]] = []
    for metric_name, metric_col in metric_map.items():
        if metric_col not in non_trap.columns:
            continue
        task_std = (
            non_trap.groupby(["Model", "Method", "ID"], dropna=False)[metric_col]
            .std()
            .dropna()
            .reset_index(name="run_std")
        )
        if task_std.empty:
            continue
        for (model, method), frame in task_std.groupby(["Model", "Method"], dropna=False):
            values = frame["run_std"].astype(float).dropna()
            rows.append(
                {
                    "Model": str(model),
                    "Method": str(method),
                    "metric": metric_name,
                    "mean_run_std": float(values.mean()) if not values.empty else np.nan,
                    "median_run_std": float(values.median()) if not values.empty else np.nan,
                    "n_tasks": int(len(values)),
                }
            )
    return pd.DataFrame(rows, columns=["Model", "Method", "metric", "mean_run_std", "median_run_std", "n_tasks"])


def generate_run_figures(evaluated_df: pd.DataFrame, question_file: str | Path, outdir: str | Path) -> list[str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titlesize": 13,
            "figure.titleweight": "normal",
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.title_fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    df = _prep_eval_df(evaluated_df, question_file)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    for old in out_path.glob("figure_*.*"):
        try:
            old.unlink()
        except Exception:
            pass
    tasktype_summary_path = out_path / "summary_tasktype_deltas.csv"
    precision_tasktype_summary_path = out_path / "summary_tasktype_deltas_atomic_precision.csv"
    hallucination_tasktype_summary_path = out_path / "summary_tasktype_deltas_hallucination_rate.csv"
    precision_risk_summary_path = out_path / "summary_precision_risk_metrics.csv"
    run_variance_summary_path = out_path / "summary_run_variance_metrics.csv"
    if tasktype_summary_path.exists():
        try:
            tasktype_summary_path.unlink()
        except Exception:
            pass
    for extra_summary_path in [
        precision_tasktype_summary_path,
        hallucination_tasktype_summary_path,
        run_variance_summary_path,
    ]:
        if extra_summary_path.exists():
            try:
                extra_summary_path.unlink()
            except Exception:
                pass
    if precision_risk_summary_path.exists():
        try:
            precision_risk_summary_path.unlink()
        except Exception:
            pass
    shapiro_values_path = out_path / "figure_7_shapiro_diagnostics_atomic_accuracy_values.csv"
    if shapiro_values_path.exists():
        try:
            shapiro_values_path.unlink()
        except Exception:
            pass
    tasktype_summary = _build_tasktype_delta_summary(df, metric_key="atomic_coverage_score")
    precision_tasktype_summary = _build_tasktype_delta_summary(df, metric_key="atomic_precision")
    hallucination_tasktype_summary = _build_tasktype_delta_summary(df, metric_key="hallucination_rate")
    if not tasktype_summary.empty:
        tasktype_summary.to_csv(tasktype_summary_path, index=False)
    if not precision_tasktype_summary.empty:
        precision_tasktype_summary.to_csv(precision_tasktype_summary_path, index=False)
    if not hallucination_tasktype_summary.empty:
        hallucination_tasktype_summary.to_csv(hallucination_tasktype_summary_path, index=False)
    shapiro_diagnostic = _build_shapiro_diagnostic_frame(df)
    if not shapiro_diagnostic.empty:
        shapiro_diagnostic.to_csv(shapiro_values_path, index=False)
    precision_risk_summary = _build_precision_risk_summary(df)
    if not precision_risk_summary.empty:
        precision_risk_summary.to_csv(precision_risk_summary_path, index=False)
    run_variance_summary = _build_run_variance_summary(df)
    if not run_variance_summary.empty:
        run_variance_summary.to_csv(run_variance_summary_path, index=False)
    _plot_model_by_category(
        df=df,
        metric_col="Atomic_Coverage_Score",
        title="Atomic Coverage Score nach Kategorie",
        outdir=out_path,
        filename="figure_1_model_category_score",
    )
    _plot_model_by_category(
        df=df,
        metric_col="Trap_Fallibility",
        title="Anfälligkeit für Fangfragen nach Kategorie",
        outdir=out_path,
        filename="figure_2_model_category_hallucination",
    )
    _plot_difficulty_aggregated(df, out_path)
    _plot_costs_separate(df, out_path)
    _plot_task_type_delta_boxplots(
        _build_tasktype_delta_frame(df, metric_key="atomic_coverage_score"),
        out_path,
        metric_key="atomic_coverage_score",
        filename="figure_5_tasktype_delta_boxplots_atomic_accuracy",
    )
    _plot_task_type_delta_heatmap(tasktype_summary, out_path)
    _plot_shapiro_diagnostics(shapiro_diagnostic, out_path)
    _plot_model_method_bar(
        precision_risk_summary,
        metric_col="Atomic_Precision",
        ylabel="Mittlere Atomic Precision",
        title="Atomic Precision nach Modell und Methode",
        outdir=out_path,
        filename="figure_8_atomic_precision_model_method",
    )
    _plot_model_method_bar(
        precision_risk_summary,
        metric_col="Hallucination_Rate",
        ylabel="Mittlere Hallucination Rate",
        title="Hallucination Rate nach Modell und Methode",
        outdir=out_path,
        filename="figure_9_hallucination_rate_model_method",
    )
    _plot_coverage_precision_scatter(precision_risk_summary, out_path)
    _plot_risk_composition(precision_risk_summary, out_path)
    _plot_task_type_delta_boxplots(
        _build_tasktype_delta_frame(df, metric_key="atomic_precision"),
        out_path,
        metric_key="atomic_precision",
        filename="figure_12_tasktype_delta_boxplots_atomic_precision",
    )
    _plot_task_type_delta_boxplots(
        _build_tasktype_delta_frame(df, metric_key="hallucination_rate"),
        out_path,
        metric_key="hallucination_rate",
        filename="figure_13_tasktype_delta_boxplots_hallucination_rate",
    )
    return [str(p) for p in sorted(out_path.glob("figure_*.*"))]
