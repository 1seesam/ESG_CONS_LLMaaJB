from __future__ import annotations

import json
import re
from typing import List, Tuple

import numpy as np


def evaluate_binary(generated: str, reference: str) -> int:
    g = str(generated).lower().strip()
    r = str(reference).lower().strip()
    if g == r:
        return 5
    pairs = [("wahr", "wahr"), ("falsch", "falsch"), ("true", "true"), ("false", "false")]
    for left, right in pairs:
        if left in r and right in g:
            return 5
    return 1


def evaluate_extraction(generated: str, reference: str) -> int:
    gen_str = str(generated).lower()
    ref_str = str(reference).lower().strip()
    return 5 if ref_str and ref_str in gen_str else 1


def evaluate_free_text_structured(
    llm_structured, question: str, generated: str, reference: str
) -> Tuple[float, str]:
    prompt = f"""
Du bist ein wissenschaftlicher ESG-Auditor.
Bewerte die Qualitaet der KANDIDATEN-ANTWORT im Vergleich zur REFERENZ.

FRAGE: {question}
REFERENZ (Gold Standard): {reference}
KANDIDATEN-ANTWORT: {generated}

Bewerte streng nach dieser Skala:
0 = Falsch / Halluzination / Thema verfehlt
1 = Starke Fehler oder Widersprueche
2 = Teilweise richtig, wichtige Details fehlen
3 = Inhaltlich korrekt, nur minimale Abweichungen
4 = Perfekt / Deckungsgleich
"""
    try:  # pragma: no cover - network dependent
        result = llm_structured.invoke(prompt)
        score = float(getattr(result, "score", np.nan))
        reason = str(getattr(result, "begruendung", ""))
        return score, reason
    except Exception as exc:
        return np.nan, f"API Error: {exc}"


def parse_atomic_facts_from_row(row) -> List[str]:
    if "Atomic_Facts" in row and str(row.get("Atomic_Facts", "")).strip():
        value = row.get("Atomic_Facts", "")
    else:
        value = row.get("Referenzantwort", "")
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


def evaluate_atomic_facts_structured(
    llm_structured,
    question: str,
    generated: str,
    atomic_facts: List[str],
    judge_prompt_template: str | None = None,
) -> Tuple[float, str, float]:
    prompt = build_atomic_facts_judge_prompt(
        question=question,
        generated=generated,
        atomic_facts=atomic_facts,
        judge_prompt_template=judge_prompt_template,
    )
    try:  # pragma: no cover - network dependent
        result = llm_structured.invoke(prompt)
        score = float(getattr(result, "score", np.nan))
        reason = str(getattr(result, "begruendung", ""))
        coverage = float(getattr(result, "coverage", np.nan))
        return score, reason, coverage
    except Exception as exc:
        return np.nan, f"API Error: {exc}", np.nan


def _build_trap_instructions(trap_mode: bool, expected_trap_behavior: str = "") -> str:
    if not trap_mode:
        return ""
    expected = str(expected_trap_behavior or "").strip()
    expected_block = (
        f"\nErwartetes Fangfragen-Verhalten:\n{expected}\n"
        if expected
        else (
            "\nErwartetes Fangfragen-Verhalten: Die Antwort erkennt die falsche oder "
            "unvollstaendige Praemisse und gibt keine unbegruendete Direktantwort.\n"
        )
    )
    return f"""
FANGFRAGEN-BEWERTUNG:
Diese Frage ist eine Fangfrage. Bewerte `trap_pass` als eigenstaendige Metrik, unabhaengig davon,
ob alle Atomic Facts vollstaendig reproduziert wurden.
- `trap_pass = 1`: Die Kandidaten-Antwort erkennt die falsche/unvollstaendige Praemisse, vermeidet eine
  unzulaessige Berechnung oder Schlussfolgerung und nennt die fachlich korrekte Einschraenkung.
- `trap_pass = 0`: Die Kandidaten-Antwort beantwortet die Fangfrage direkt, uebernimmt die falsche
  Praemisse, erfindet fehlende Angaben oder liefert eine unzulaessige Berechnung/Schlussfolgerung.
- Eine kurze korrekte Zurueckweisung darf `trap_pass = 1` erhalten, auch wenn einzelne Gold-Facts fehlen.
{expected_block}
""".strip()


def build_atomic_facts_judge_prompt(
    question: str,
    generated: str,
    atomic_facts: List[str],
    judge_prompt_template: str | None = None,
    trap_mode: bool = False,
    expected_trap_behavior: str = "",
) -> str:
    facts_block = "\n".join([f"- {fact}" for fact in atomic_facts]) if atomic_facts else "- (keine Facts)"
    trap_instructions = _build_trap_instructions(trap_mode, expected_trap_behavior)
    default_prompt = f"""
Du bist ein wissenschaftlicher ESG-Auditor.
Bewerte die KANDIDATEN-ANTWORT ausschliesslich gegen die Atomic Facts.

FRAGE: {question}
ATOMIC FACTS (Goldstandard):
{facts_block}

KANDIDATEN-ANTWORT:
{generated}

Aufgabe:
1) Beurteile, wie viele Facts inhaltlich korrekt abgedeckt sind.
2) Vergib dann einen strengen Likert-Score:
0 = stark falsch/halluziniert
1 = viele wesentliche Facts fehlen oder falsch
2 = teilweise korrekt
3 = weitgehend korrekt, kleine Luecken
4 = deckungsgleich zu den Facts

{trap_instructions}
"""
    prompt = default_prompt
    if judge_prompt_template:
        try:
            prompt = judge_prompt_template.format(
                question=question,
                generated=generated,
                facts_block=facts_block,
                trap_instructions=trap_instructions,
                expected_trap_behavior=str(expected_trap_behavior or "").strip(),
                is_trap_question=int(bool(trap_mode)),
            )
        except Exception:
            prompt = default_prompt
    if trap_instructions and "trap_pass" not in prompt.lower():
        prompt = prompt.rstrip() + "\n\n" + trap_instructions
    return prompt


def parse_judge_json(text: str) -> Tuple[float, str, float]:
    raw = str(text or "").strip()
    if not raw:
        return np.nan, "Judge returned empty response", np.nan
    payload = None
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        score_match = re.search(r'"?score"?\s*:\s*([0-4](?:\.\d+)?)', raw, flags=re.IGNORECASE)
        coverage_match = re.search(r'"?coverage"?\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)', raw, flags=re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            coverage = float(coverage_match.group(1)) if coverage_match else np.nan
            return score, f"Judge JSON partly recovered: {raw[:220]}", coverage
        return np.nan, f"Judge JSON parse failed: {raw[:240]}", np.nan
    score = payload.get("score", np.nan)
    reason = payload.get("begruendung", "")
    coverage = payload.get("coverage", np.nan)
    try:
        score = float(score)
    except Exception:
        score = np.nan
    if not np.isnan(score) and (score < 0 or score > 4):
        score = np.nan
    try:
        coverage = float(coverage)
    except Exception:
        coverage = np.nan
    return score, str(reason), coverage


def parse_claim_labels_json(text: str) -> Tuple[int, int, int, float]:
    raw = str(text or "").strip()
    if not raw:
        return 0, 0, 0, float("nan")
    payload = None
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    def _to_int(value) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

    def _recover_int_field(name: str) -> int:
        match = re.search(rf'"?{re.escape(name)}"?\s*:\s*(\d+)', raw, flags=re.IGNORECASE)
        return _to_int(match.group(1)) if match else 0

    def _coerce_trap_pass(value) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        text_value = str(value).strip().lower()
        if text_value in {"true", "yes", "ja", "pass", "passed"}:
            return 1.0
        if text_value in {"false", "no", "nein", "fail", "failed"}:
            return 0.0
        try:
            numeric = float(value)
        except Exception:
            return float("nan")
        if np.isnan(numeric):
            return float("nan")
        return 1.0 if numeric >= 0.5 else 0.0

    def _recover_trap_pass() -> float:
        match = re.search(
            r'"?trap_pass"?\s*:\s*(true|false|0|1|0\.0|1\.0)',
            raw,
            flags=re.IGNORECASE,
        )
        return _coerce_trap_pass(match.group(1)) if match else float("nan")

    if not isinstance(payload, dict):
        supported = _recover_int_field("supported")
        contradicted = _recover_int_field("contradicted")
        not_in_scope = _recover_int_field("not_in_scope")
        return supported, contradicted, not_in_scope, _recover_trap_pass()

    supported = _to_int(payload.get("supported", 0))
    contradicted = _to_int(payload.get("contradicted", 0))
    not_in_scope = _to_int(payload.get("not_in_scope", 0))
    trap_pass = payload.get("trap_pass", float("nan"))
    trap_pass = _coerce_trap_pass(trap_pass)
    return supported, contradicted, not_in_scope, trap_pass


def parse_predicted_claim_labels_json(text: str) -> Tuple[int, int, int]:
    raw = str(text or "").strip()
    if not raw:
        return 0, 0, 0
    payload = None
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None

    def _to_int(value) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

    def _recover_int_field(name: str) -> int:
        match = re.search(rf'"?{re.escape(name)}"?\s*:\s*(\d+)', raw, flags=re.IGNORECASE)
        return _to_int(match.group(1)) if match else 0

    if not isinstance(payload, dict):
        supported = _recover_int_field("predicted_supported_claims")
        unsupported = _recover_int_field("predicted_unsupported_claims")
        total = _recover_int_field("predicted_total_claims")
    else:
        supported = _to_int(payload.get("predicted_supported_claims", 0))
        unsupported = _to_int(payload.get("predicted_unsupported_claims", 0))
        total = _to_int(payload.get("predicted_total_claims", 0))
    if total <= 0 and (supported + unsupported) > 0:
        total = supported + unsupported
    return supported, unsupported, total


def is_trap_question(answer_type: str, reference_answer: str) -> bool:
    marker_text = f"{answer_type} {reference_answer}".lower()
    markers = (
        "fangfrage",
        "trap",
        "false premise",
        "falsche prämisse",
        "falsche praemisse",
    )
    return any(marker in marker_text for marker in markers)


def classify_answer_type(answer_type: str) -> str:
    typ = str(answer_type).lower()
    if "wahr" in typ or "falsch" in typ:
        return "binary"
    if "extraktion" in typ or "extract" in typ:
        return "extraction"
    if "atomic" in typ or "fakt" in typ:
        return "atomic_facts"
    return "free_text"


def sanitize_judge_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_confidence_scores(generated: str) -> List[int]:
    text = str(generated or "")
    scores: List[int] = []
    for match in re.finditer(r"#\s*([0-3])\b", text, flags=re.IGNORECASE):
        try:
            value = int(match.group(1))
        except Exception:
            continue
        if 0 <= value <= 3:
            scores.append(value)
    return scores


def extract_confidence_mean(generated: str) -> Tuple[float, int]:
    scores = extract_confidence_scores(generated)
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), 1


def derive_likert_from_claim_counts(supported: int, contradicted: int, not_in_scope: int) -> float:
    total = int(supported) + int(contradicted) + int(not_in_scope)
    if total <= 0:
        return float("nan")
    support_rate = supported / total
    contradiction_rate = contradicted / total
    if support_rate == 1.0 and contradiction_rate == 0.0:
        return 4.0
    if support_rate >= 0.75 and contradiction_rate == 0.0:
        return 3.0
    if support_rate >= 0.5:
        return 2.0
    if support_rate > 0.0:
        return 1.0
    return 0.0
