from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_model_id(model_id: str) -> str:
    return str(model_id or "").strip().lower()


def _fetch_openrouter_models(api_key: str | None):
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_pricing_map(models_payload) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for item in models_payload.get("data", []):
        model_id = str(item.get("id", "")).strip()
        if not model_id:
            continue
        pricing = item.get("pricing", {}) or {}
        # OpenRouter exposes prompt/completion pricing in USD per token.
        in_price = _to_float(pricing.get("prompt", 0.0), 0.0)
        out_price = _to_float(pricing.get("completion", 0.0), 0.0)
        out[model_id] = {"input": in_price, "output": out_price}
    return out


def resolve_pricing_for_models(
    models: List[str],
    default_pricing: Dict[str, Dict[str, float]],
    cache_path: str = "./eval_data/pricing_cache.json",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    source = "fallback_default"
    fetched_map: Dict[str, Dict[str, float]] = {}
    fetch_error = ""

    try:
        payload = _fetch_openrouter_models(api_key)
        fetched_map = _extract_pricing_map(payload)
        source = "openrouter_live"
        _write_json(
            cache_path,
            {
                "fetched_at_unix": int(time.time()),
                "source": source,
                "pricing_map": fetched_map,
            },
        )
    except Exception as exc:
        fetch_error = str(exc)
        cache = _read_json(cache_path) or {}
        cached_map = cache.get("pricing_map", {}) if isinstance(cache, dict) else {}
        if isinstance(cached_map, dict) and cached_map:
            fetched_map = cached_map
            source = "openrouter_cache"

    normalized_fetched_map: Dict[str, Dict[str, float]] = {}
    for fetched_key, pricing in fetched_map.items():
        norm_key = _normalize_model_id(fetched_key)
        if not norm_key:
            continue
        normalized_fetched_map.setdefault(norm_key, pricing)
        base_key = norm_key.split(":", 1)[0]
        normalized_fetched_map.setdefault(base_key, pricing)
        if "/" in base_key:
            providerless_key = base_key.split("/", 1)[-1]
            normalized_fetched_map.setdefault(providerless_key, pricing)

    normalized_default_pricing: Dict[str, Dict[str, float]] = {}
    for default_key, pricing in default_pricing.items():
        norm_default_key = _normalize_model_id(default_key)
        if not norm_default_key:
            continue
        normalized_default_pricing.setdefault(norm_default_key, pricing)
        default_base_key = norm_default_key.split(":", 1)[0]
        normalized_default_pricing.setdefault(default_base_key, pricing)
        if "/" in default_base_key:
            normalized_default_pricing.setdefault(default_base_key.split("/", 1)[-1], pricing)

    final: Dict[str, Dict[str, float]] = {}
    details: Dict[str, object] = {
        "source": source,
        "cache_path": cache_path,
        "fetch_error": fetch_error,
        "resolved_models": {},
    }
    for model in models:
        model_key = str(model).strip()
        normalized_model_key = _normalize_model_id(model_key)
        base_model_key = normalized_model_key.split(":", 1)[0]
        providerless_model_key = base_model_key.split("/", 1)[-1]

        if normalized_model_key in normalized_fetched_map:
            pricing = normalized_fetched_map[normalized_model_key]
            final[model] = pricing
            details["resolved_models"][model] = {
                "source": source,
                "pricing": pricing,
            }
        elif base_model_key in normalized_fetched_map:
            pricing = normalized_fetched_map[base_model_key]
            final[model] = pricing
            details["resolved_models"][model] = {
                "source": f"{source}_matched_base_id",
                "pricing": pricing,
                "matched_model_id": base_model_key,
            }
        elif providerless_model_key in normalized_fetched_map:
            pricing = normalized_fetched_map[providerless_model_key]
            final[model] = pricing
            details["resolved_models"][model] = {
                "source": f"{source}_matched_providerless_id",
                "pricing": pricing,
                "matched_model_id": providerless_model_key,
            }
        elif normalized_model_key in normalized_default_pricing:
            pricing = normalized_default_pricing[normalized_model_key]
            final[model] = pricing
            details["resolved_models"][model] = {
                "source": "fallback_default",
                "pricing": pricing,
            }
        elif base_model_key in normalized_default_pricing:
            pricing = normalized_default_pricing[base_model_key]
            final[model] = pricing
            details["resolved_models"][model] = {
                "source": "fallback_default_matched_base_id",
                "pricing": pricing,
                "matched_model_id": base_model_key,
            }
        elif providerless_model_key in normalized_default_pricing:
            pricing = normalized_default_pricing[providerless_model_key]
            final[model] = pricing
            details["resolved_models"][model] = {
                "source": "fallback_default_matched_providerless_id",
                "pricing": pricing,
                "matched_model_id": providerless_model_key,
            }
        else:
            final[model] = {"input": 0.0, "output": 0.0}
            details["resolved_models"][model] = {
                "source": "missing_unknown_default_zero",
                "pricing": {"input": 0.0, "output": 0.0},
            }
    return final, details
