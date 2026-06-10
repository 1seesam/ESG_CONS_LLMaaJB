from __future__ import annotations

import os
import json
import re
from dataclasses import asdict
from typing import Dict, List

from .config_schema import QuestionItem
from .io import read_json


REQUIRED_FIELDS = ["id", "question"]


def parse_atomic_facts(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(v).strip() for v in payload if str(v).strip()]
        except Exception:
            pass
    parts = re.split(r"\r?\n|;|\|", text)
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


def validate_questions(raw_items: List[Dict[str, object]]) -> List[str]:
    errors: List[str] = []
    for idx, item in enumerate(raw_items):
        for field in REQUIRED_FIELDS:
            if not str(item.get(field, "")).strip():
                errors.append(f"row {idx}: missing '{field}'")
        if not isinstance(item, dict):
            errors.append(f"row {idx}: must be object")
    return errors


def normalize_question(item: Dict[str, object], idx: int) -> QuestionItem:
    qid = str(item.get("id", idx)).strip()
    atomic_facts = parse_atomic_facts(item.get("atomic_facts", ""))
    if not atomic_facts:
        fact_cols = []
        for key, value in item.items():
            if str(key).lower().startswith("atomic_fact_"):
                cleaned = str(value).strip()
                if cleaned:
                    fact_cols.append(cleaned)
        atomic_facts = fact_cols
    return QuestionItem(
        id=qid,
        question=str(item.get("question", "")).strip(),
        category=str(item.get("category", "")).strip(),
        task_type_raw=str(item.get("task_type_raw", "")).strip(),
        reference_answer=str(item.get("reference_answer", "")).strip(),
        atomic_facts=atomic_facts,
        answer_type=str(item.get("answer_type", "")).strip(),
        expected_trap_behavior=str(item.get("expected_trap_behavior", "")).strip(),
        max_words=str(item.get("max_words", "")).strip(),
        relevant_docs=str(item.get("relevant_docs", "")).strip(),
        difficulty=str(item.get("difficulty", "")).strip(),
    )


def is_extraction_item(item: Dict[str, object]) -> bool:
    answer_type = str(item.get("answer_type", "")).strip().lower()
    category = str(item.get("category", "")).strip().lower()
    extraction_markers = ("extraktion", "extraction")
    return any(marker in answer_type for marker in extraction_markers) or any(
        marker in category for marker in extraction_markers
    )


def load_questions(path: str) -> List[QuestionItem]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions file not found: {path}")
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError("Questions file must contain a JSON list.")
    errors = validate_questions(raw)
    if errors:
        raise ValueError("Invalid questions file:\n" + "\n".join(errors[:20]))
    filtered = [item for item in raw if not is_extraction_item(item)]
    return [normalize_question(item, idx) for idx, item in enumerate(filtered)]


def questions_to_rows(items: List[QuestionItem]) -> List[Dict[str, object]]:
    return [asdict(item) for item in items]
