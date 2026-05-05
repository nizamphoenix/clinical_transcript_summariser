"""
Template-aware validator for generated samples.

A sample is valid iff:
  1. Outer JSON parses.
  2. Has top-level "transcript" (str) and the correct label key for the template.
  3. Label matches the template schema (keys + types).
  4. Every populated evidence_span is a verbatim substring of the transcript.

Returns (sample_dict | None, reason_str).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_wrappers(raw: str) -> str:
    """Remove markdown fences and prose preambles around a JSON object."""
    raw = raw.strip()
    fence = _FENCE_RE.match(raw)
    if fence:
        raw = fence.group(1).strip()
    match = _OBJECT_RE.search(raw)
    if match:
        return match.group(0)
    return raw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_and_validate(raw: str, template_name: str) -> tuple[dict | None, str]:
    """Parse raw model output and validate against the given template schema."""
    from src.data_generation.templates import REGISTRY

    entry = REGISTRY[template_name]
    label_key = entry["label_key"]

    raw = _strip_wrappers(raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json_parse_error: {e.msg}"

    if not isinstance(obj, dict):
        return None, "top_level_not_object"

    transcript = obj.get("transcript")
    label = obj.get(label_key)

    if not isinstance(transcript, str) or not transcript.strip():
        return None, "missing_or_empty_transcript"
    if not isinstance(label, dict):
        return None, f"missing_or_invalid_{label_key}"

    # Dispatch to template-specific schema check
    schema_err = _SCHEMA_CHECKERS[str(label_key)](label)  # type: ignore[index]
    if schema_err:
        return None, f"schema: {schema_err}"

    span_err = _check_evidence_spans(transcript, label)
    if span_err:
        return None, f"ungrounded_span: {span_err}"

    return {"transcript": transcript, label_key: label}, "ok"


# ---------------------------------------------------------------------------
# SOAP schema checker
# ---------------------------------------------------------------------------

_SOAP_TOP = {"subjective", "objective", "assessment", "plan"}
_SOAP_SUBJ = {"chief_complaint", "hpi", "pmh", "meds", "allergies"}


def _check_soap(soap: dict) -> str | None:
    if not _SOAP_TOP.issubset(soap.keys()):
        return f"missing top keys {_SOAP_TOP - soap.keys()}"

    subj = soap["subjective"]
    if not isinstance(subj, dict):
        return "subjective not dict"
    if not _SOAP_SUBJ.issubset(subj.keys()):
        return f"missing subjective keys {_SOAP_SUBJ - subj.keys()}"

    for scalar in ("chief_complaint", "hpi"):
        node = subj[scalar]
        if (
            not isinstance(node, dict)
            or "text" not in node
            or "evidence_span" not in node
        ):
            return f"subjective.{scalar} malformed"

    for lst in ("pmh", "meds", "allergies"):
        if not isinstance(subj[lst], list):
            return f"subjective.{lst} not list"
        for item in subj[lst]:
            if (
                not isinstance(item, dict)
                or "text" not in item
                or "evidence_span" not in item
            ):
                return f"subjective.{lst} item malformed"

    exam = soap["objective"].get("exam")
    if (
        not isinstance(exam, dict)
        or "text" not in exam
        or "evidence_span" not in exam
    ):
        return "objective.exam malformed"

    problems = soap["assessment"].get("problems")
    if not isinstance(problems, list):
        return "assessment.problems not list"
    for p in problems:
        if (
            not isinstance(p, dict)
            or "name" not in p
            or "evidence_span" not in p
        ):
            return "assessment.problems item malformed"

    items = soap["plan"].get("items")
    if not isinstance(items, list):
        return "plan.items not list"
    for it in items:
        if (
            not isinstance(it, dict)
            or "action" not in it
            or "evidence_span" not in it
        ):
            return "plan.items item malformed"

    return None


# ---------------------------------------------------------------------------
# REFERRAL schema checker
# ---------------------------------------------------------------------------

_REFERRAL_SCALARS = {"specialty", "patient", "reason", "request"}
_REFERRAL_LISTS = {"history", "current_meds"}


def _check_referral(ref: dict) -> str | None:
    for key in _REFERRAL_SCALARS:
        if key not in ref:
            return f"missing key: {key}"
        node = ref[key]
        if (
            not isinstance(node, dict)
            or "text" not in node
            or "evidence_span" not in node
        ):
            return f"{key} malformed"

    for key in _REFERRAL_LISTS:
        if key not in ref:
            return f"missing key: {key}"
        if not isinstance(ref[key], list):
            return f"{key} not list"
        for item in ref[key]:
            if (
                not isinstance(item, dict)
                or "text" not in item
                or "evidence_span" not in item
            ):
                return f"{key} item malformed"

    return None


# ---------------------------------------------------------------------------
# MSE schema checker
# ---------------------------------------------------------------------------

_MSE_DOMAINS = {
    "appearance",
    "behaviour",
    "speech",
    "mood",
    "affect",
    "thought",
    "cognition",
    "insight",
}


def _check_mse(mse: dict) -> str | None:
    for domain in _MSE_DOMAINS:
        if domain not in mse:
            return f"missing domain: {domain}"
        node = mse[domain]
        if (
            not isinstance(node, dict)
            or "text" not in node
            or "evidence_span" not in node
        ):
            return f"{domain} malformed"
    return None


# ---------------------------------------------------------------------------
# Registry + shared helpers
# ---------------------------------------------------------------------------

_SCHEMA_CHECKERS = {
    "soap": _check_soap,
    "referral": _check_referral,
    "mse": _check_mse,
}


def _check_evidence_spans(transcript: str, label: Any) -> str | None:
    for span in _collect_spans(label):
        if span is None or span == "":
            continue
        if not isinstance(span, str):
            return f"non-string span: {span!r}"
        if span not in transcript:
            preview = span if len(span) <= 80 else span[:77] + "..."
            return f"not in transcript: {preview!r}"
    return None


def _collect_spans(node: Any) -> list:
    spans: list = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "evidence_span":
                spans.append(v)
            else:
                spans.extend(_collect_spans(v))
    elif isinstance(node, list):
        for item in node:
            spans.extend(_collect_spans(item))
    return spans
