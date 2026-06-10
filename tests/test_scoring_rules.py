from src.benchmarking.scoring import (
    build_atomic_facts_judge_prompt,
    classify_answer_type,
    evaluate_binary,
    evaluate_extraction,
    parse_atomic_facts_from_row,
    parse_claim_labels_json,
    parse_judge_json,
    parse_predicted_claim_labels_json,
)


def test_evaluate_binary_true_match():
    assert evaluate_binary("Wahr", "Wahr") == 5
    assert evaluate_binary("Falsch", "Wahr") == 1


def test_evaluate_extraction_contains_reference():
    assert evaluate_extraction("Der Wert ist 145 kWh", "145") == 5
    assert evaluate_extraction("Der Wert ist 200 kWh", "145") == 1


def test_classify_answer_type():
    assert classify_answer_type("Wahr / Falsch") == "binary"
    assert classify_answer_type("Extraktion") == "extraction"
    assert classify_answer_type("Freitext") == "free_text"


def test_parse_atomic_facts_from_row_json():
    row = {"Atomic_Facts": '["F1","F2"]', "Referenzantwort": ""}
    assert parse_atomic_facts_from_row(row) == ["F1", "F2"]


def test_parse_atomic_facts_from_row_fallback():
    row = {"Atomic_Facts": "", "Referenzantwort": "F1; F2"}
    assert parse_atomic_facts_from_row(row) == ["F1", "F2"]


def test_parse_judge_json_recovers_score_from_truncated_json():
    score, reason, coverage = parse_judge_json('```json { "score": 1, "begruendung": "abgeschnitten')
    assert score == 1
    assert "partly recovered" in reason


def test_trap_prompt_defines_trap_pass_independently_from_coverage():
    prompt = build_atomic_facts_judge_prompt(
        question="Fangfrage?",
        generated="Nein, das ist nicht berechenbar.",
        atomic_facts=["Keine Berechnung moeglich"],
        trap_mode=True,
        expected_trap_behavior="Keine Direktberechnung liefern.",
    )
    assert "trap_pass = 1" in prompt
    assert "trap_pass = 0" in prompt
    assert "unabhaengig" in prompt
    assert "Keine Direktberechnung liefern." in prompt


def test_parse_claim_labels_json_parses_and_recovers_trap_pass():
    supported, contradicted, not_in_scope, trap_pass = parse_claim_labels_json(
        '{"supported": 2, "contradicted": 0, "not_in_scope": 1, "trap_pass": true}'
    )
    assert (supported, contradicted, not_in_scope, trap_pass) == (2, 0, 1, 1.0)

    supported, contradicted, not_in_scope, trap_pass = parse_claim_labels_json(
        '```json { "supported": 1, "contradicted": 0, "not_in_scope": 2, "trap_pass": 1'
    )
    assert (supported, contradicted, not_in_scope, trap_pass) == (1, 0, 2, 1.0)


def test_parse_predicted_claim_labels_json_parses_and_recovers_counts():
    supported, unsupported, total = parse_predicted_claim_labels_json(
        '{"predicted_supported_claims": 3, "predicted_unsupported_claims": 2, "predicted_total_claims": 5}'
    )
    assert (supported, unsupported, total) == (3, 2, 5)

    supported, unsupported, total = parse_predicted_claim_labels_json(
        '```json { "predicted_supported_claims": 1, "predicted_unsupported_claims": 2'
    )
    assert (supported, unsupported, total) == (1, 2, 3)
