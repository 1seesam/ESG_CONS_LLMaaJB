import pandas as pd

from src.benchmarking.evaluation_pipeline import (
    _cached_evaluation_is_reusable,
    _judge_prompt_hash,
)


def test_cached_evaluation_requires_matching_prompt_hash():
    prompt_template = "FRAGE: {question}\nFACTS:\n{facts_block}\nANTWORT:\n{generated}\n{trap_instructions}"
    prompt_hash = _judge_prompt_hash(prompt_template)
    row = pd.Series(
        {
            "Judge_Begruendung": "ok",
            "judge_model": "openai/gpt-4o-mini",
            "judge_prompt_version": "trap_v2",
            "judge_prompt_hash": prompt_hash,
            "Likert_Score": 4.0,
        }
    )

    assert _cached_evaluation_is_reusable(
        row,
        judge_model="openai/gpt-4o-mini",
        judge_prompt_version="trap_v2",
        judge_prompt_hash=prompt_hash,
    )

    assert not _cached_evaluation_is_reusable(
        row.drop(labels=["judge_prompt_hash"]),
        judge_model="openai/gpt-4o-mini",
        judge_prompt_version="trap_v2",
        judge_prompt_hash=prompt_hash,
    )

    assert not _cached_evaluation_is_reusable(
        row,
        judge_model="openai/gpt-4o-mini",
        judge_prompt_version="trap_v2",
        judge_prompt_hash="different",
    )
