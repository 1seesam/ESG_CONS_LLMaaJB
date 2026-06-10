import json
from pathlib import Path

from src.benchmarking.pricing_provider import resolve_pricing_for_models


def test_resolve_pricing_for_models_matches_providerless_model_name(monkeypatch):
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_pricing_provider"
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": [
            {
                "id": "meta-llama/llama-4-scout",
                "pricing": {"prompt": 8e-08, "completion": 3e-07},
            }
        ]
    }

    monkeypatch.setattr(
        "src.benchmarking.pricing_provider._fetch_openrouter_models",
        lambda api_key: payload,
    )

    pricing_map, details = resolve_pricing_for_models(
        models=["llama-4-scout"],
        default_pricing={},
        cache_path=str(base / "pricing_cache_providerless.json"),
    )

    assert pricing_map["llama-4-scout"] == {"input": 8e-08, "output": 3e-07}
    assert details["resolved_models"]["llama-4-scout"]["source"] == "openrouter_live"


def test_resolve_pricing_for_models_matches_base_model_id_with_suffix(monkeypatch):
    base = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_pricing_provider"
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": [
            {
                "id": "meta-llama/llama-4-scout:free",
                "pricing": {"prompt": 8e-08, "completion": 3e-07},
            }
        ]
    }

    monkeypatch.setattr(
        "src.benchmarking.pricing_provider._fetch_openrouter_models",
        lambda api_key: payload,
    )

    pricing_map, details = resolve_pricing_for_models(
        models=["meta-llama/llama-4-scout"],
        default_pricing={},
        cache_path=str(base / "pricing_cache_suffix.json"),
    )

    assert pricing_map["meta-llama/llama-4-scout"] == {"input": 8e-08, "output": 3e-07}
    assert details["resolved_models"]["meta-llama/llama-4-scout"]["source"] == "openrouter_live"
