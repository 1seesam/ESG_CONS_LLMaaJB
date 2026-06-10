from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping

import pandas as pd


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return read_excel(path)


def write_excel(path: str, df: pd.DataFrame) -> None:
    ensure_parent(path)
    df.to_excel(path, index=False)


def dataframe_from_rows(rows: List[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def file_exists(path: str) -> bool:
    return os.path.exists(path)

