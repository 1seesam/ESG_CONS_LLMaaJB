from src.benchmarking.prompting import build_prompt


def test_max_words_numeric_rule():
    prompt = build_prompt("Zero-Shot", "Frage?", max_words="25")
    assert "Max 25 Woerter" in prompt


def test_max_words_text_rule():
    prompt = build_prompt("Zero-Shot", "Frage?", max_words="nur den Buchstaben der korrekten Antwort")
    assert "Zusatzregel zur Laenge" in prompt
    assert "nur den Buchstaben der korrekten Antwort" in prompt

