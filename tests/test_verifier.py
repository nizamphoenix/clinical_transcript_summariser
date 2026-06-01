"""
Unit tests for src/verifier.py.

All tests use hand-built fixtures: no model inference, no file I/O.
The SOAP template is used throughout because it is the simplest schema
to construct manually.
"""

import json

import pytest

from src.verifier import (
    content_overlap,
    grounding_counts,
    null_classification,
    parse_prediction,
    reward,
    score_prediction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRANSCRIPT = (
    "Patient: I woke up with a terrible headache this morning, really sharp pain. "
    "No history of migraines but my BP was a bit high last time. "
    "HCP: BP is 145 over 90 today. Sounds like tension headache. Try panadol and rest."
)

# A well-formed, fully-populated SOAP prediction grounded in TRANSCRIPT above.
PERFECT_SOAP = {
    "subjective": {
        "chief_complaint": {
            "text": "headache",
            "evidence_span": "terrible headache this morning",
        },
        "hpi": {
            "text": "sharp headache started this morning",
            "evidence_span": "really sharp pain",
        },
        "pmh": [
            {
                "text": "high blood pressure",
                "evidence_span": "BP was a bit high last time",
            }
        ],
        "meds": [],
        "allergies": [],
    },
    "objective": {
        "exam": {
            "text": "BP 145/90",
            "evidence_span": "BP is 145 over 90 today",
        }
    },
    "assessment": {
        "problems": [
            {
                "name": "tension headache",
                "evidence_span": "Sounds like tension headache",
            }
        ]
    },
    "plan": {
        "items": [
            {
                "action": "take panadol and rest",
                "evidence_span": "Try panadol and rest",
            }
        ]
    },
}

# Same shape as PERFECT_SOAP but all text/name/action values are empty strings.
# evidence_spans are still grounded so schema_valid stays 1.
ALL_NULL_SOAP = {
    "subjective": {
        "chief_complaint": {"text": "", "evidence_span": ""},
        "hpi": {"text": "", "evidence_span": ""},
        "pmh": [],
        "meds": [],
        "allergies": [],
    },
    "objective": {"exam": {"text": "", "evidence_span": ""}},
    "assessment": {"problems": []},
    "plan": {"items": []},
}

# A prediction where one evidence_span is NOT in the transcript.
HALLUCINATED_SOAP = json.loads(json.dumps(PERFECT_SOAP))  # deep copy
HALLUCINATED_SOAP["subjective"]["chief_complaint"]["evidence_span"] = (
    "invented phrase xyz"
)

# A prediction where gold-null fields (meds, allergies) are filled with invented text.
# schema_valid stays 1 because shape is correct; spans are grounded.
OVER_POPULATED_SOAP = json.loads(json.dumps(PERFECT_SOAP))  # deep copy
OVER_POPULATED_SOAP["subjective"]["meds"] = [
    {"text": "aspirin 100mg daily", "evidence_span": "aspirin 100mg daily"}
]
OVER_POPULATED_SOAP["subjective"]["allergies"] = [
    {"text": "penicillin", "evidence_span": "penicillin"}
]

# A prediction missing the 'plan' top-level key — schema invalid.
SCHEMA_INVALID_SOAP = {
    "subjective": PERFECT_SOAP["subjective"],
    "objective": PERFECT_SOAP["objective"],
    "assessment": PERFECT_SOAP["assessment"],
    # 'plan' intentionally omitted
}


# ---------------------------------------------------------------------------
# parse_prediction
# ---------------------------------------------------------------------------


def test_parse_prediction_plain_json():
    raw = json.dumps(PERFECT_SOAP)
    result = parse_prediction(raw)
    assert result == PERFECT_SOAP


def test_parse_prediction_fenced():
    raw = "```json\n" + json.dumps(PERFECT_SOAP) + "\n```"
    result = parse_prediction(raw)
    assert result == PERFECT_SOAP


def test_parse_prediction_garbage():
    assert parse_prediction("not json at all") is None


# ---------------------------------------------------------------------------
# content_overlap
# ---------------------------------------------------------------------------


def test_content_overlap_perfect():
    score = content_overlap(PERFECT_SOAP, PERFECT_SOAP)
    assert score == pytest.approx(1.0)


def test_content_overlap_all_null():
    score = content_overlap(ALL_NULL_SOAP, PERFECT_SOAP)
    assert score == pytest.approx(0.0)


def test_content_overlap_partial():
    # A prediction that shares some but not all content with gold.
    partial = json.loads(json.dumps(PERFECT_SOAP))
    partial["subjective"]["chief_complaint"]["text"] = "headache"
    partial["subjective"]["hpi"]["text"] = ""  # empty out one field
    score = content_overlap(partial, PERFECT_SOAP)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# grounding_counts
# ---------------------------------------------------------------------------


def test_grounding_counts_all_grounded():
    grounded, total = grounding_counts(PERFECT_SOAP, TRANSCRIPT)
    assert total > 0
    assert grounded == total


def test_grounding_counts_hallucination():
    grounded, total = grounding_counts(HALLUCINATED_SOAP, TRANSCRIPT)
    assert total > 0
    assert grounded < total


def test_grounding_counts_no_spans():
    grounded, total = grounding_counts(ALL_NULL_SOAP, TRANSCRIPT)
    assert total == 0
    assert grounded == 0


# ---------------------------------------------------------------------------
# null_classification
# ---------------------------------------------------------------------------


def test_null_classification_perfect():
    stats = null_classification(PERFECT_SOAP, PERFECT_SOAP)
    assert stats["wrong_null"] == 0
    assert stats["gold_filled"] > 0


def test_null_classification_all_null():
    stats = null_classification(ALL_NULL_SOAP, PERFECT_SOAP)
    # Every field gold filled should be reported as a miss.
    assert stats["wrong_null"] == stats["gold_filled"]


def test_null_classification_none_pred():
    stats = null_classification(None, PERFECT_SOAP)
    assert stats["wrong_null"] == stats["gold_filled"]


# ---------------------------------------------------------------------------
# score_prediction
# ---------------------------------------------------------------------------


def test_score_prediction_perfect():
    raw = json.dumps(PERFECT_SOAP)
    s = score_prediction("soap", raw, PERFECT_SOAP, TRANSCRIPT)
    assert s["schema_valid"] == 1
    assert s["content_overlap"] == pytest.approx(1.0)
    assert s["wrong_null"] == 0
    assert s["grounded_spans"] == s["total_spans"]


def test_score_prediction_all_null():
    raw = json.dumps(ALL_NULL_SOAP)
    s = score_prediction("soap", raw, PERFECT_SOAP, TRANSCRIPT)
    assert s["schema_valid"] == 1
    assert s["content_overlap"] == pytest.approx(0.0)
    assert s["wrong_null"] == s["gold_filled"]


def test_score_prediction_schema_invalid():
    raw = json.dumps(SCHEMA_INVALID_SOAP)
    s = score_prediction("soap", raw, PERFECT_SOAP, TRANSCRIPT)
    assert s["schema_valid"] == 0


def test_score_prediction_unparseable():
    s = score_prediction("soap", "not json", PERFECT_SOAP, TRANSCRIPT)
    assert s["parse"] == 0
    assert s["schema_valid"] == 0


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------


def _make_score(
    schema_valid,
    content_ov,
    wrong_null,
    gold_filled,
    grounded,
    total_spans,
    over_populated=0,
    gold_null=1,
):
    return {
        "schema_valid": schema_valid,
        "content_overlap": content_ov,
        "wrong_null": wrong_null,
        "gold_filled": gold_filled,
        "grounded_spans": grounded,
        "total_spans": total_spans,
        "over_populated": over_populated,
        "gold_null": gold_null,
    }


def test_reward_schema_invalid_is_negative_one():
    s = _make_score(0, 0.9, 0, 5, 5, 5)
    assert reward(s) == pytest.approx(-1.0)


def test_reward_perfect():
    s = _make_score(1, 1.0, 0, 5, 5, 5)
    r = reward(s)
    assert r > 0.5


def test_reward_all_null():
    s = _make_score(1, 0.0, 5, 5, 0, 0)
    r = reward(s)
    assert r < 0.0


def test_reward_ordering():
    """perfect > partial > all_null > schema_invalid"""
    perfect = reward(_make_score(1, 1.0, 0, 5, 5, 5))
    partial = reward(_make_score(1, 0.4, 2, 5, 3, 5))
    all_null = reward(_make_score(1, 0.0, 5, 5, 0, 0))
    invalid = reward(_make_score(0, 0.9, 0, 5, 5, 5))

    assert perfect > partial > all_null > invalid


def test_reward_over_populated_penalised():
    """perfect > miss > hallucination_via_over_populate > schema_invalid"""
    # Perfect: high overlap, no misses, no over-population, all grounded.
    perfect = reward(
        _make_score(1, 1.0, 0, 5, 5, 5, over_populated=0, gold_null=3)
    )
    # Miss: moderate overlap, 1 wrong_null, no over-population.
    miss = reward(
        _make_score(1, 0.6, 1, 5, 5, 5, over_populated=0, gold_null=3)
    )
    # Over-populated: same overlap as miss but fills 3 gold-null fields.
    over_pop = reward(
        _make_score(1, 0.6, 0, 5, 5, 5, over_populated=3, gold_null=3)
    )
    # Schema invalid: hard gate.
    invalid = reward(_make_score(0, 0.9, 0, 5, 5, 5))

    assert perfect > miss
    assert miss > over_pop
    assert over_pop > invalid


def test_null_classification_over_populated():
    """Fields gold left empty but pred fills should count as over_populated."""
    # OVER_POPULATED_SOAP fills meds/allergies which are empty lists in gold (PERFECT_SOAP).
    # Empty lists contribute no leaves, so over_populated comes from extra pred leaves only.
    stats = null_classification(OVER_POPULATED_SOAP, PERFECT_SOAP)
    assert stats["over_populated"] > 0
