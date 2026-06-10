import math

from src.benchmarking.scoring import extract_confidence_mean, extract_confidence_scores


def test_extract_confidence_scores_parses_af_hash_format():
    scores = extract_confidence_scores("Fact A #2; Fact B #0; Fact C #3")
    assert scores == [2, 0, 3]


def test_extract_confidence_scores_ignores_named_formats_without_hash_markers():
    scores = extract_confidence_scores("Sicherheitslevel: 2\nSicherheit: 1\nUnsicherheit: 0")
    assert scores == []


def test_extract_confidence_mean_returns_nan_when_missing():
    mean, found = extract_confidence_mean("Keine parsebare Sicherheit enthalten.")
    assert math.isnan(mean)
    assert found == 0


def test_extract_confidence_mean_averages_detected_scores():
    mean, found = extract_confidence_mean("Fact A #3\nFact B #0\nFact C #3")
    assert math.isclose(mean, 2.0, rel_tol=1e-6)
    assert found == 1
