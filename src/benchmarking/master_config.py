from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

ALLOWED_METHODS = {"Zero-Shot", "Advanced-Prompt", "RAG"}


DEFAULT_MASTER_CONFIG: Dict[str, Any] = {
    "catalog_excel": "./eval_data/Fragenkatalog_MA.xlsx",
    "auto_convert_questions": True,
    "input_file": "./eval_data/questions.json",
    "benchmark_output": "./eval_data/results_benchmark_final.xlsx",
    "evaluated_output": "./eval_data/results_evaluated_scientific.xlsx",
    "num_runs": 5,
    "seed": 42,
    "temperature": 0.0,
    "throttle_seconds": 0.5,
    "max_retries": 2,
    "use_rag": True,
    "judge_model": "openai/gpt-4o-mini",
    "judge_prompt_version": "trap_v2",
    "judge_prompt_path": "./prompts/judge_atomic_facts.txt",
    "judge_throttle_seconds": 4.0,
    "judge_max_retries": 5,
    "baseline_method": "Zero-Shot",
    "time_cost_factor": 0.002,
    "runs_root": "./eval_data/runs",
    "run_uuid": "",
    "default_max_words": "",
    "auto_reindex": True,
    "rag_chunk_size": 400,
    "rag_chunk_overlap": 70,
    "max_rag_context_chars": 6000,
    "max_prompt_chars": 14000,
    "slow_call_seconds": 10.0,
    "max_consecutive_api_fails": 10,
    "parallel_workers": 1,
    "judge_parallel_workers": 1,
    "auto_resume_until_complete": True,
    "max_auto_resume_attempts": 5,
    "auto_resume_sleep_seconds": 30.0,
    "prompt_zero_shot_path": "./prompts/zero_shot.txt",
    "prompt_advanced_path": "./prompts/advanced_prompt.txt",
    "prompt_rag_path": "./prompts/rag_prompt.txt",
    "enable_overconfidence": False,
    "enable_visualizations": False,
    "enable_robustness": False,
    "methods": ["Zero-Shot", "Advanced-Prompt", "RAG"],
    "models": ["google/gemini-2.0-flash-exp:free"],
    "model_temperatures": {},
}


def _to_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "ja"}:
        return True
    if text in {"0", "false", "no", "n", "nein"}:
        return False
    return default


def _normalize_number_text(value: Any) -> str:
    text = str(value).strip().replace(" ", "").replace("_", "")
    if "," in text and "." in text:
        # If comma appears after dot, interpret as decimal comma (e.g. 1.234,56).
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")
        # Otherwise treat comma as thousands separator (e.g. 1,234.56).
        return text.replace(",", "")
    if "," in text:
        parts = text.split(",")
        if len(parts) > 1 and parts[0].lstrip("+-").isdigit() and all(p.isdigit() and len(p) == 3 for p in parts[1:]):
            # 6,000 -> 6000
            return "".join(parts)
        # 0,5 -> 0.5
        return text.replace(",", ".")
    if "." in text:
        parts = text.split(".")
        if len(parts) > 1 and parts[0].lstrip("+-").isdigit() and all(p.isdigit() and len(p) == 3 for p in parts[1:]):
            # 6.000 -> 6000
            return "".join(parts)
    return text


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(_normalize_number_text(value))


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(_to_float(value))


def _typed_value(key: str, value: Any):
    if pd.isna(value):
        if key in {"run_uuid", "input_file", "benchmark_output", "evaluated_output"}:
            return ""
        if key == "default_max_words":
            return None
        return value
    if key in {
        "num_runs",
        "seed",
        "max_retries",
        "judge_max_retries",
        "rag_chunk_size",
        "rag_chunk_overlap",
        "max_rag_context_chars",
        "max_prompt_chars",
        "max_consecutive_api_fails",
        "max_auto_resume_attempts",
        "parallel_workers",
        "judge_parallel_workers",
    }:
        return _to_int(value)
    if key in {
        "temperature",
        "throttle_seconds",
        "judge_throttle_seconds",
        "time_cost_factor",
        "slow_call_seconds",
        "auto_resume_sleep_seconds",
    }:
        return _to_float(value)
    if key in {
        "use_rag",
        "auto_reindex",
        "auto_convert_questions",
        "enable_overconfidence",
        "enable_visualizations",
        "enable_robustness",
        "auto_resume_until_complete",
    }:
        return _to_bool(value)
    if key == "default_max_words":
        text = str(value).strip()
        if not text:
            return None
        try:
            return _to_int(text)
        except ValueError:
            return None
    if key in {
        "run_uuid",
        "input_file",
        "benchmark_output",
        "evaluated_output",
        "judge_model",
        "judge_prompt_version",
        "judge_prompt_path",
        "baseline_method",
        "runs_root",
        "prompt_zero_shot_path",
        "prompt_advanced_path",
        "prompt_rag_path",
    }:
        return str(value).strip()
    return value


def load_master_config(path: str) -> Dict[str, Any]:
    cfg = dict(DEFAULT_MASTER_CONFIG)
    if not path or not os.path.exists(path):
        return cfg

    settings_df = pd.read_excel(path, sheet_name="settings")
    for _, row in settings_df.iterrows():
        key = str(row.get("key", "")).strip()
        if not key:
            continue
        value = row.get("value", "")
        cfg[key] = _typed_value(key, value)

    models_df = pd.read_excel(path, sheet_name="models")
    models: List[str] = []
    model_temperatures: Dict[str, float] = {}
    for _, row in models_df.iterrows():
        model = str(row.get("model", "")).strip()
        if not model:
            continue
        enabled = _to_bool(row.get("enabled", True), default=True)
        if not enabled:
            continue
        models.append(model)
        temp_value = str(row.get("temperature", "")).strip()
        if temp_value:
            try:
                model_temperatures[model] = _to_float(temp_value)
            except ValueError:
                pass
    if models:
        cfg["models"] = models
    cfg["model_temperatures"] = model_temperatures

    methods_df = pd.read_excel(path, sheet_name="methods")
    methods: List[str] = []
    for _, row in methods_df.iterrows():
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        if method not in ALLOWED_METHODS:
            continue
        enabled = _to_bool(row.get("enabled", True), default=True)
        if enabled:
            methods.append(method)
    if methods:
        cfg["methods"] = methods

    return cfg


def create_master_config_template(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    settings_rows = [
        ("input_file", DEFAULT_MASTER_CONFIG["input_file"]),
        ("catalog_excel", DEFAULT_MASTER_CONFIG["catalog_excel"]),
        ("auto_convert_questions", DEFAULT_MASTER_CONFIG["auto_convert_questions"]),
        ("benchmark_output", DEFAULT_MASTER_CONFIG["benchmark_output"]),
        ("evaluated_output", DEFAULT_MASTER_CONFIG["evaluated_output"]),
        ("num_runs", DEFAULT_MASTER_CONFIG["num_runs"]),
        ("seed", DEFAULT_MASTER_CONFIG["seed"]),
        ("temperature", DEFAULT_MASTER_CONFIG["temperature"]),
        ("throttle_seconds", DEFAULT_MASTER_CONFIG["throttle_seconds"]),
        ("max_retries", DEFAULT_MASTER_CONFIG["max_retries"]),
        ("use_rag", DEFAULT_MASTER_CONFIG["use_rag"]),
        ("judge_model", DEFAULT_MASTER_CONFIG["judge_model"]),
        ("judge_prompt_version", DEFAULT_MASTER_CONFIG["judge_prompt_version"]),
        ("judge_prompt_path", DEFAULT_MASTER_CONFIG["judge_prompt_path"]),
        ("judge_throttle_seconds", DEFAULT_MASTER_CONFIG["judge_throttle_seconds"]),
        ("judge_max_retries", DEFAULT_MASTER_CONFIG["judge_max_retries"]),
        ("baseline_method", DEFAULT_MASTER_CONFIG["baseline_method"]),
        ("time_cost_factor", DEFAULT_MASTER_CONFIG["time_cost_factor"]),
        ("runs_root", DEFAULT_MASTER_CONFIG["runs_root"]),
        ("run_uuid", DEFAULT_MASTER_CONFIG["run_uuid"]),
        ("default_max_words", DEFAULT_MASTER_CONFIG["default_max_words"]),
        ("auto_reindex", DEFAULT_MASTER_CONFIG["auto_reindex"]),
        ("rag_chunk_size", DEFAULT_MASTER_CONFIG["rag_chunk_size"]),
        ("rag_chunk_overlap", DEFAULT_MASTER_CONFIG["rag_chunk_overlap"]),
        ("max_rag_context_chars", DEFAULT_MASTER_CONFIG["max_rag_context_chars"]),
        ("max_prompt_chars", DEFAULT_MASTER_CONFIG["max_prompt_chars"]),
        ("slow_call_seconds", DEFAULT_MASTER_CONFIG["slow_call_seconds"]),
        ("max_consecutive_api_fails", DEFAULT_MASTER_CONFIG["max_consecutive_api_fails"]),
        ("parallel_workers", DEFAULT_MASTER_CONFIG["parallel_workers"]),
        ("judge_parallel_workers", DEFAULT_MASTER_CONFIG["judge_parallel_workers"]),
        ("auto_resume_until_complete", DEFAULT_MASTER_CONFIG["auto_resume_until_complete"]),
        ("max_auto_resume_attempts", DEFAULT_MASTER_CONFIG["max_auto_resume_attempts"]),
        ("auto_resume_sleep_seconds", DEFAULT_MASTER_CONFIG["auto_resume_sleep_seconds"]),
        ("prompt_zero_shot_path", DEFAULT_MASTER_CONFIG["prompt_zero_shot_path"]),
        ("prompt_advanced_path", DEFAULT_MASTER_CONFIG["prompt_advanced_path"]),
        ("prompt_rag_path", DEFAULT_MASTER_CONFIG["prompt_rag_path"]),
        ("enable_overconfidence", DEFAULT_MASTER_CONFIG["enable_overconfidence"]),
        ("enable_visualizations", DEFAULT_MASTER_CONFIG["enable_visualizations"]),
        ("enable_robustness", DEFAULT_MASTER_CONFIG["enable_robustness"]),
    ]
    settings_df = pd.DataFrame(settings_rows, columns=["key", "value"])

    models_df = pd.DataFrame(
        [
            {"model": "openai/gpt-4o-mini", "enabled": True, "temperature": 0.0},
            {"model": "google/gemini-2.0-flash-exp:free", "enabled": False, "temperature": 0.0},
        ]
    )
    methods_df = pd.DataFrame(
        [
            {"method": "Zero-Shot", "enabled": True},
            {"method": "Advanced-Prompt", "enabled": True},
            {"method": "RAG", "enabled": True},
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        settings_df.to_excel(writer, sheet_name="settings", index=False)
        models_df.to_excel(writer, sheet_name="models", index=False)
        methods_df.to_excel(writer, sheet_name="methods", index=False)
