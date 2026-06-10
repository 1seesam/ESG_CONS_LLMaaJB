from __future__ import annotations

import time
from typing import Dict


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing: Dict[str, Dict[str, float]],
) -> float:
    if model_name not in pricing:
        return 0.0
    p = pricing[model_name]
    # Pricing values are interpreted as USD per token.
    return (input_tokens * p.get("input", 0.0)) + (output_tokens * p.get("output", 0.0))


def invoke_with_retry(llm, prompt: str, max_retries: int, throttle_seconds: float):
    last_error = ""
    for attempt in range(max_retries + 1):
        started = time.time()
        try:
            response = llm.invoke(prompt)
            usage = response.usage_metadata or {}
            return {
                "content": str(response.content),
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "duration_sec": round(time.time() - started, 3),
                "error": "",
            }
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = str(exc)
            time.sleep(throttle_seconds * (attempt + 1))

    return {
        "content": f"### API ERROR ###: {last_error}",
        "input_tokens": 0,
        "output_tokens": 0,
        "duration_sec": 0.0,
        "error": last_error,
    }
