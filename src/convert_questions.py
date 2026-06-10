import argparse
import json
import os
import re

import pandas as pd


TRAP_QUESTION_OVERRIDES = {
    "40": {
        "expected_trap_behavior": (
            "Verneint eine abschliessende GEG-Einhaltung; benennt, dass der Primaerenergiebedarf "
            "allein nicht ausreicht und weitere Nachweise bzw. Vergleichswerte erforderlich sind."
        ),
    },
    "41": {
        "reference_answer": (
            "Nein; Eine Luft-Luft-Waermepumpe hat kein wasserfuehrendes bzw. hydraulisches Rohrsystem; "
            "Ein hydraulischer Abgleich des Rohrsystems ist daher nicht zwingend erforderlich bzw. nicht sinnvoll moeglich."
        ),
        "expected_trap_behavior": (
            "Verneint die Pflicht zum hydraulischen Abgleich; idealerweise mit Hinweis, dass bei einer "
            "Luft-Luft-Waermepumpe kein wasserfuehrendes/hydraulisches Rohrsystem abgeglichen wird."
        ),
    },
    "42": {
        "expected_trap_behavior": (
            "Lehnt eine Berechnung der zuzukaufenden Strommenge ab; benennt, dass Gesamtstrombedarf "
            "und/oder Eigenverbrauch fehlen und daher keine belastbare Berechnung moeglich ist."
        ),
    },
    "43": {
        "expected_trap_behavior": (
            "Lehnt eine sichere Austauschpflicht aus dem Gebaeudebaujahr allein ab; benennt, dass "
            "Alter und Art der konkreten Heizungsanlage sowie Ausnahmen relevant sind."
        ),
    },
    "44": {
        "reference_answer": (
            "Das GEIG wurde nie zurueckgezogen; Das GEIG und dessen Verordnung sind aktuell noch in Kraft; "
            "Es gibt daher keinen Rueckzugszeitpunkt und keinen Rueckzugsgrund."
        ),
        "expected_trap_behavior": (
            "Weist die Praemisse zurueck, dass das GEIG zurueckgezogen wurde; nennt keinen erfundenen "
            "Zeitpunkt oder Grund, sondern dass es weiterhin in Kraft ist."
        ),
    },
    "45": {
        "reference_answer": (
            "Heizoel erzeugt direkte Emissionen im Objekt; Direkte Emissionen sind Scope-1-Emissionen; "
            "100 kWh/m2a * 0,3 kgCO2/kWh = 30 kgCO2/m2a waere Scope 1, nicht Scope 2; "
            "Scope-2-Emissionen sind aus den Angaben nicht berechenbar."
        ),
        "expected_trap_behavior": (
            "Weist die Scope-2-Falle zurueck; berechnet nicht 30 kgCO2/m2a als Scope 2, sondern "
            "erklaert, dass dieser Wert Scope 1 waere und Scope 2 aus den Angaben nicht berechenbar ist."
        ),
    },
}


def _extract_atomic_facts(row: pd.Series):
    direct_cols = ["Atomic Facts", "Atomic_Facts", "AtomicFacts"]
    for col in direct_cols:
        if col in row.index and str(row[col]).strip():
            text = str(row[col]).strip()
            parts = re.split(r"\r?\n|;|\|", text)
            facts = [p.strip(" -\t") for p in parts if p.strip(" -\t")]
            if facts:
                return facts

    reference_cols = ["Referenzantwort", "Goldstandard"]
    for col in reference_cols:
        if col in row.index and str(row[col]).strip():
            text = str(row[col]).strip()
            parts = text.split(";")
            facts = [p.strip(" -\t") for p in parts if p.strip(" -\t")]
            if facts:
                return facts

    facts = []
    for key in row.index:
        if str(key).lower().startswith("atomic_fact_"):
            value = str(row[key]).strip()
            if value:
                facts.append(value)
    return facts


def _derive_expected_trap_behavior(item: dict) -> str:
    answer_type = str(item.get("answer_type", "")).strip().lower()
    existing = str(item.get("expected_trap_behavior", "")).strip()
    if existing:
        return existing
    if "fangfrage" not in answer_type and "trap" not in answer_type:
        return ""
    reference = str(item.get("reference_answer", "")).strip()
    if reference:
        return (
            "Die Antwort soll die falsche oder unvollstaendige Praemisse erkennen, "
            "keine Direktberechnung bzw. unbegruendete Schlussfolgerung liefern und "
            f"sinngemaess diese Einschraenkung nennen: {reference}"
        )
    return "Die Antwort soll die Fangfrage erkennen und fehlende oder falsche Voraussetzungen benennen."


def _apply_trap_question_overrides(item: dict) -> None:
    qid = str(item.get("id", "")).strip()
    overrides = TRAP_QUESTION_OVERRIDES.get(qid)
    if not overrides:
        return
    item.update(overrides)
    if "reference_answer" in overrides:
        item["atomic_facts"] = _split_reference_into_facts(overrides["reference_answer"])


def _split_reference_into_facts(reference_answer: str) -> list[str]:
    parts = re.split(r"\r?\n|;|\|", str(reference_answer or ""))
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


def excel_to_json(excel_path="./eval_data/Fragenkatalog_MA.xlsx", json_path="./eval_data/questions.json"):
    if not os.path.exists(excel_path):
        print(f"FEHLER: Datei '{excel_path}' nicht gefunden!")
        return

    print(f"Lese Excel: {excel_path} ...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as exc:
        print(f"CRITICAL ERROR: Konnte Excel nicht lesen. {exc}")
        return

    column_mapping = {
        "ID": "id",
        "Kategorie": "category",
        "Aufgabentyp": "task_type_raw",
        "Frage": "question",
        "Goldstandard": "reference_answer",
        "Referenzantwort": "reference_answer",
        "Context | Dokumente": "relevant_docs",
        "Schwierigkeitslevel": "difficulty",
        "Antwortoptionen": "answer_type",
        "Expected Trap Behavior": "expected_trap_behavior",
        "Expected_Trap_Behavior": "expected_trap_behavior",
        "Trap-Kriterium": "expected_trap_behavior",
        "Fangfrage Kriterium": "expected_trap_behavior",
        "Fangfragen-Kriterium": "expected_trap_behavior",
        "max wörter": "max_words",
        "max_wörter": "max_words",
        "Atomic Facts": "atomic_facts",
        "Atomic_Facts": "atomic_facts",
    }

    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str)

    records = []
    for _, row in df.iterrows():
        item = {}
        for src_col, target_col in column_mapping.items():
            if src_col in row.index:
                item[target_col] = str(row[src_col]).strip()
        facts = _extract_atomic_facts(row)
        item["atomic_facts"] = facts
        if not str(item.get("reference_answer", "")).strip() and facts:
            item["reference_answer"] = "; ".join(facts)
        item["expected_trap_behavior"] = _derive_expected_trap_behavior(item)
        _apply_trap_question_overrides(item)
        records.append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    print(f"OK: {len(records)} Fragen konvertiert.")
    print("   Feld 'atomic_facts' wird als Liste geschrieben.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert question catalog Excel to questions.json")
    parser.add_argument("--excel-path", default="./eval_data/Fragenkatalog_MA.xlsx")
    parser.add_argument("--json-path", default="./eval_data/questions.json")
    args = parser.parse_args()
    excel_to_json(excel_path=args.excel_path, json_path=args.json_path)
