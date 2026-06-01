"""
Verifier: schema validation, content scoring, and DPO reward for clinical transcript extraction.

This module is the single source of truth used by:
  - eval notebooks (score_prediction)
  - v3_pair_gen (score_prediction + reward for pair ranking)
  - v3_dpo_train (reward as preference signal)

Reuses schema checkers and span utilities from src.data_generation.validate
rather than duplicating them.
"""

from __future__ import annotations

import json
from typing import Any

from src.data_generation.validate import (
    _check_mse,
    _check_referral,
    _check_soap,
    _collect_spans,
    _strip_wrappers,
)

# ---------------------------------------------------------------------------
# Schema checker registry (template name -> checker function)
# ---------------------------------------------------------------------------

SCHEMA_CHECKERS: dict[str, Any] = {
    "soap": _check_soap,
    "referral_a": _check_referral,
    "referral_b": _check_referral,
    "mse": _check_mse,
}

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_prediction(raw: str) -> dict | None:
    """Strip markdown fences and parse the first JSON object from raw model output."""
    try:
        return json.loads(_strip_wrappers(raw))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Content-leaf traversal
# Handles the three evidence-bearing shapes our schemas use:
#   {"text": ..., "evidence_span": ...}
#   {"name": ..., "evidence_span": ...}
#   {"action": ..., "evidence_span": ...}
# ---------------------------------------------------------------------------


def _walk_content_leaves(
    node: Any, path: tuple = ()
) -> list[tuple[tuple, Any]]:
    results: list[tuple[tuple, Any]] = []
    if isinstance(node, dict):
        if {"text", "evidence_span"}.issubset(node.keys()):
            results.append((path + ("text",), node.get("text")))
            return results
        if {"name", "evidence_span"}.issubset(node.keys()):
            results.append((path + ("name",), node.get("name")))
            return results
        if {"action", "evidence_span"}.issubset(node.keys()):
            results.append((path + ("action",), node.get("action")))
            return results
        for key, value in node.items():
            results.extend(_walk_content_leaves(value, path + (key,)))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            results.extend(_walk_content_leaves(item, path + (idx,)))
    return results


def _is_filled(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _content_token_set(node: Any) -> set[str]:
    tokens: set[str] = set()
    for _, value in _walk_content_leaves(node):
        if isinstance(value, str) and value.strip():
            tokens.update(value.lower().split())
    return tokens


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------


def content_overlap(pred: dict | None, gold: dict) -> float:
    """Jaccard overlap of content tokens between prediction and gold."""
    pred_tokens = _content_token_set(pred) if isinstance(pred, dict) else set()
    gold_tokens = _content_token_set(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    return len(pred_tokens & gold_tokens) / max(
        len(pred_tokens | gold_tokens), 1
    )


def grounding_counts(pred: dict | None, transcript: str) -> tuple[int, int]:
    """Return (grounded_span_count, total_span_count).

    A span is grounded if it appears verbatim in the transcript.
    """
    if not isinstance(pred, dict):
        return 0, 0
    spans = [s for s in _collect_spans(pred) if s]
    grounded = sum(1 for s in spans if isinstance(s, str) and s in transcript)
    return grounded, len(spans)


def null_classification(pred: dict | None, gold: dict) -> dict:
    """Compare prediction vs gold at matching content-leaf paths.

    Returns:
        gold_filled     : fields gold populated
        gold_null       : fields gold left empty
        wrong_null      : gold filled, pred null/missing (a miss)
        correct_null    : gold null, pred null/missing (correctly empty)
        over_populated  : gold null, pred filled (potential hallucination)
    """
    pred_leaves: dict[tuple, Any] = {}
    if isinstance(pred, dict):
        pred_leaves = {p: v for p, v in _walk_content_leaves(pred)}

    gold_filled = gold_null = wrong_null = correct_null = over_populated = 0
    seen: set[tuple] = set()

    for path, gold_val in _walk_content_leaves(gold):
        seen.add(path)
        gold_is_filled = _is_filled(gold_val)
        pred_is_filled = _is_filled(pred_leaves.get(path))
        if gold_is_filled:
            gold_filled += 1
            if not pred_is_filled:
                wrong_null += 1
        else:
            gold_null += 1
            if pred_is_filled:
                over_populated += 1
            else:
                correct_null += 1

    for path, pred_val in pred_leaves.items():
        if path not in seen and _is_filled(pred_val):
            over_populated += 1

    return {
        "gold_filled": gold_filled,
        "gold_null": gold_null,
        "wrong_null": wrong_null,
        "correct_null": correct_null,
        "over_populated": over_populated,
    }


def _is_all_null(pred: dict | None) -> bool:
    if not isinstance(pred, dict):
        return True
    return all(not _is_filled(v) for _, v in _walk_content_leaves(pred))


def score_prediction(
    template: str,
    raw: str,
    gold: dict,
    transcript: str,
) -> dict:
    """Score a single raw model output against gold.

    Returns a flat dict of all verifier terms. The 'pred_label' key holds
    the parsed prediction (or None) for downstream use.
    """
    pred = parse_prediction(raw)
    parse_ok = isinstance(pred, dict)
    schema_err = SCHEMA_CHECKERS[template](pred) if parse_ok else "unparseable"
    grounded, total_spans = grounding_counts(pred, transcript)
    null_stats = null_classification(pred, gold)
    return {
        "parse": int(parse_ok),
        "schema_valid": int(parse_ok and schema_err is None),
        "content_overlap": content_overlap(pred, gold) if parse_ok else 0.0,
        "grounded_spans": grounded,
        "total_spans": total_spans,
        "all_null": int(_is_all_null(pred)),
        "pred_label": pred,
        **null_stats,
    }


# ---------------------------------------------------------------------------
# DPO reward scalar
# ---------------------------------------------------------------------------


def reward(score: dict) -> float:
    """Scalar reward for DPO pair ranking.

    schema_valid is a hard gate: any schema-invalid output scores -1.0.
    Among valid outputs:
        reward = content_overlap
                 - 0.5 * wrong_null_rate
                 - 0.5 * ungrounded_span_rate
    """
    if not score["schema_valid"]:
        return -1.0
    gold_filled = score.get("gold_filled") or 1
    total_spans = score.get("total_spans", 0)
    wrong_null_rate = score["wrong_null"] / gold_filled
    ungrounded_rate = (
        (1.0 - score["grounded_spans"] / total_spans) if total_spans else 0.0
    )
    return (
        score["content_overlap"] - 0.5 * wrong_null_rate - 0.5 * ungrounded_rate
    )
