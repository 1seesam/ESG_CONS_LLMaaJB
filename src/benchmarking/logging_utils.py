from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_uuid() -> str:
    local_now = datetime.now().astimezone()
    timestamp = local_now.strftime("%Y-%m-%d_%H-%M")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Dict[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return sha256_text(canonical)


def get_git_commit(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", cwd, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def init_run_artifacts(runs_root: str, run_uuid: str | None = None) -> Dict[str, str]:
    run_uuid = str(run_uuid or default_run_uuid())
    run_dir = os.path.join(runs_root, run_uuid)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_uuid": run_uuid,
        "run_dir": run_dir,
        "raw_results_jsonl": os.path.join(run_dir, "raw_results.jsonl"),
        "evaluated_results_jsonl": os.path.join(run_dir, "evaluated_results.jsonl"),
        "stats_report_json": os.path.join(run_dir, "stats_report.json"),
        "summary_main_tests_csv": os.path.join(run_dir, "summary_main_tests.csv"),
        "summary_trap_questions_csv": os.path.join(run_dir, "summary_trap_questions.csv"),
        "summary_robustness_csv": os.path.join(run_dir, "summary_robustness.csv"),
        "summary_overconfidence_csv": os.path.join(run_dir, "summary_overconfidence.csv"),
        "pricing_snapshot": os.path.join(run_dir, "pricing.snapshot.json"),
        "config_snapshot": os.path.join(run_dir, "config.snapshot.json"),
        "run_log": os.path.join(run_dir, "run.log"),
    }


def setup_run_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
