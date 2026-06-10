import json
from pathlib import Path

import pytest

from src.benchmarking.dataset import load_questions


def _test_dir():
    path = Path(__file__).resolve().parents[1] / "tmp" / "pytest_local_dataset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_questions_requires_fields():
    path = _test_dir() / "questions.json"
    path.write_text(json.dumps([{"id": "1"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(str(path))


def test_load_questions_success():
    path = _test_dir() / "questions_ok.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "Ist das gueltig?",
                    "reference_answer": "Wahr",
                    "atomic_facts": ["Fakt A", "Fakt B"],
                    "answer_type": "Wahr / Falsch",
                }
            ]
        ),
        encoding="utf-8",
    )
    items = load_questions(str(path))
    assert len(items) == 1
    assert items[0].id == "1"
    assert items[0].question == "Ist das gueltig?"
    assert items[0].atomic_facts == ["Fakt A", "Fakt B"]


def test_load_questions_atomic_facts_from_columns():
    path = _test_dir() / "questions_atomic_cols.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "2",
                    "question": "Nenne Fakten",
                    "atomic_fact_1": "A",
                    "atomic_fact_2": "B",
                }
            ]
        ),
        encoding="utf-8",
    )
    items = load_questions(str(path))
    assert items[0].atomic_facts == ["A", "B"]
