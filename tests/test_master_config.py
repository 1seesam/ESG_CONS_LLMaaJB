from pathlib import Path

import pandas as pd

from src.benchmarking.master_config import (
    create_master_config_template,
    load_master_config,
)


def test_create_and_load_master_config():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_master_config"
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "master_config.xlsx"

    create_master_config_template(str(cfg_path))
    cfg = load_master_config(str(cfg_path))

    assert cfg_path.exists()
    assert "models" in cfg and cfg["models"]
    assert "methods" in cfg and cfg["methods"]
    assert isinstance(cfg.get("model_temperatures", {}), dict)
    assert "temperature" in cfg
    assert cfg.get("rag_chunk_size") == 400
    assert cfg.get("rag_chunk_overlap") == 70
    assert cfg.get("enable_overconfidence") is False
    assert cfg.get("enable_visualizations") is False
    assert cfg.get("enable_robustness") is False
    assert cfg["methods"] == ["Zero-Shot", "Advanced-Prompt", "RAG"]


def test_run_uuid_empty_not_nan():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_master_config_nan"
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "master_config.xlsx"
    create_master_config_template(str(cfg_path))
    cfg = load_master_config(str(cfg_path))
    assert cfg.get("run_uuid", "") in {"", None}


def test_number_like_strings_are_parsed_from_excel():
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_master_config_number_strings"
    base.mkdir(parents=True, exist_ok=True)
    cfg_path = base / "master_config.xlsx"
    create_master_config_template(str(cfg_path))

    settings_df = pd.read_excel(cfg_path, sheet_name="settings")
    settings_df.loc[settings_df["key"] == "max_rag_context_chars", "value"] = "6,000"
    settings_df.loc[settings_df["key"] == "throttle_seconds", "value"] = "0,5"
    settings_df.loc[settings_df["key"] == "default_max_words", "value"] = "1.200"
    settings_df.loc[settings_df["key"] == "rag_chunk_size", "value"] = "800"
    settings_df.loc[settings_df["key"] == "rag_chunk_overlap", "value"] = "120"

    models_df = pd.read_excel(cfg_path, sheet_name="models")
    methods_df = pd.read_excel(cfg_path, sheet_name="methods")
    models_df.loc[0, "temperature"] = "0,7"

    with pd.ExcelWriter(cfg_path, engine="openpyxl") as writer:
        settings_df.to_excel(writer, sheet_name="settings", index=False)
        models_df.to_excel(writer, sheet_name="models", index=False)
        methods_df.to_excel(writer, sheet_name="methods", index=False)

    cfg = load_master_config(str(cfg_path))
    assert cfg["max_rag_context_chars"] == 6000
    assert cfg["throttle_seconds"] == 0.5
    assert cfg["default_max_words"] == 1200
    assert cfg["rag_chunk_size"] == 800
    assert cfg["rag_chunk_overlap"] == 120
    assert cfg["model_temperatures"][cfg["models"][0]] == 0.7
