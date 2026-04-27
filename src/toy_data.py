"""
Toy data generator for the v1 spike.

Produces short, messy clinician-patient transcripts with speaker tags
("HCP:" / "Patient:" on new lines), fillers ("uh", "um", "like"), and
ungrammatical fragments. Each transcript is paired with a SOAP JSON target
whose `evidence_span` fields are exact substrings of the transcript.

Design:
- 15 transcript-shape templates (functions) for linguistic variety.
- 8 slot pools (complaints, durations, meds, allergies, pmh, exam,
  assessment, plan).
- Each shape uses a subset of slots; unused fields become null/[].
- Deterministic via random.Random(seed); raises if it can't find n unique
  transcripts.

Invariant
---------
For every leaf in the SOAP target that has both `text`/`name`/`action` and
`evidence_span`, the evidence_span is an exact substring of the transcript.
This is the property the model is being trained to reproduce.
"""

from __future__ import annotations

import random
from typing import Callable

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

COMPLAINTS = [
    "chest pain",
    "headache",
    "back pain",
    "sore throat",
    "cough",
    "shortness of breath",
    "abdominal pain",
    "dizziness",
    "nausea",
    "rash",
]
DURATIONS = [
    "2 days",
    "a week",
    "since last night",
    "3 hours",
    "since yesterday",
    "couple of weeks",
]
MEDS = [
    "aspirin",
    "panadol",
    "metformin",
    "atorvastatin",
    "ventolin",
    "ibuprofen",
    "lisinopril",
    "amoxicillin",
]
ALLERGIES = [
    "penicillin",
    "nuts",
    "shellfish",
    "latex",
    "none",
]
PMH = [
    "hypertension",
    "diabetes",
    "asthma",
    "high cholesterol",
    "depression",
    "no significant history",
]
EXAMS = [
    "BP 140 over 90",
    "chest clear",
    "throat red",
    "abdomen soft",
    "no swelling noted",
    "looks well",
]
ASSESSMENTS = [
    "viral infection",
    "tension headache",
    "muscle strain",
    "uncontrolled hypertension",
    "upper respiratory infection",
    "gastritis",
]
PLANS = [
    "rest and fluids",
    "return if worse",
    "panadol as needed",
    "blood test next week",
    "follow up in 2 weeks",
    "refer to physio",
]


# ---------------------------------------------------------------------------
# Helpers for building SOAP dicts
# ---------------------------------------------------------------------------


def _leaf(text: str, span: str) -> dict:
    return {"text": text, "evidence_span": span}


def _empty_soap() -> dict:
    return {
        "subjective": {
            "chief_complaint": {"text": None, "evidence_span": None},
            "hpi": {"text": None, "evidence_span": None},
            "pmh": [],
            "meds": [],
            "allergies": [],
        },
        "objective": {
            "exam": {"text": None, "evidence_span": None},
        },
        "assessment": {"problems": []},
        "plan": {"items": []},
    }


# ---------------------------------------------------------------------------
# 15 transcript-shape templates
#
# Each function signature: shape(slots: dict) -> (transcript: str, soap: dict)
# Slots dict keys: complaint, duration, med, allergy, pmh, exam, assessment,
# plan. Each shape uses a subset; unused are ignored.
# ---------------------------------------------------------------------------


def shape01(s):
    t = (
        f"HCP: what brings you in today\n"
        f"Patient: uh {s['complaint']} since like {s['duration']}\n"
        f"HCP: any meds\n"
        f"Patient: {s['med']} um yeah\n"
        f"HCP: allergies\n"
        f"Patient: {s['allergy']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    hpi = f"{s['complaint']} since like {s['duration']}"
    soap["subjective"]["hpi"] = _leaf(hpi, hpi)
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    if s["allergy"] != "none":
        soap["subjective"]["allergies"] = [_leaf(s["allergy"], s["allergy"])]
    return t, soap


def shape02(s):
    t = (
        f"Patient: doc i got {s['complaint']}\n"
        f"HCP: how long\n"
        f"Patient: {s['duration']} now\n"
        f"HCP: ok plan is {s['plan']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["hpi"] = _leaf(s["duration"], s["duration"])
    soap["plan"]["items"] = [{"action": s["plan"], "evidence_span": s["plan"]}]
    return t, soap


def shape03(s):
    t = (
        f"HCP: history\n"
        f"Patient: i have {s['pmh']}\n"
        f"HCP: whats wrong today\n"
        f"Patient: um {s['complaint']}\n"
        f"HCP: meds\n"
        f"Patient: just {s['med']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    if s["pmh"] != "no significant history":
        soap["subjective"]["pmh"] = [_leaf(s["pmh"], s["pmh"])]
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    return t, soap


def shape04(s):
    t = (
        f"HCP: how can i help\n"
        f"Patient: {s['complaint']} for {s['duration']}\n"
        f"HCP: on exam {s['exam']}\n"
        f"HCP: looks like {s['assessment']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    hpi = f"{s['complaint']} for {s['duration']}"
    soap["subjective"]["hpi"] = _leaf(hpi, hpi)
    soap["objective"]["exam"] = _leaf(s["exam"], s["exam"])
    soap["assessment"]["problems"] = [
        {"name": s["assessment"], "evidence_span": s["assessment"]}
    ]
    return t, soap


def shape05(s):
    t = (
        f"Patient: hey doc\n"
        f"HCP: hi whats up\n"
        f"Patient: like {s['complaint']} you know\n"
        f"HCP: since when\n"
        f"Patient: {s['duration']}\n"
        f"HCP: any allergies\n"
        f"Patient: {s['allergy']}\n"
        f"HCP: plan {s['plan']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["hpi"] = _leaf(s["duration"], s["duration"])
    if s["allergy"] != "none":
        soap["subjective"]["allergies"] = [_leaf(s["allergy"], s["allergy"])]
    soap["plan"]["items"] = [{"action": s["plan"], "evidence_span": s["plan"]}]
    return t, soap


def shape06(s):
    t = (
        f"HCP: meds list\n"
        f"Patient: {s['med']}\n"
        f"HCP: ok and complaint\n"
        f"Patient: uh {s['complaint']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    return t, soap


def shape07(s):
    t = (
        f"Patient: i think i got {s['complaint']}\n"
        f"HCP: pmh\n"
        f"Patient: {s['pmh']}\n"
        f"HCP: examining now {s['exam']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    if s["pmh"] != "no significant history":
        soap["subjective"]["pmh"] = [_leaf(s["pmh"], s["pmh"])]
    soap["objective"]["exam"] = _leaf(s["exam"], s["exam"])
    return t, soap


def shape08(s):
    t = (
        f"HCP: tell me whats going on\n"
        f"Patient: well um {s['complaint']} mostly\n"
        f"HCP: how long\n"
        f"Patient: {s['duration']}\n"
        f"HCP: assessment {s['assessment']} plan {s['plan']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["hpi"] = _leaf(s["duration"], s["duration"])
    soap["assessment"]["problems"] = [
        {"name": s["assessment"], "evidence_span": s["assessment"]}
    ]
    soap["plan"]["items"] = [{"action": s["plan"], "evidence_span": s["plan"]}]
    return t, soap


def shape09(s):
    t = (
        f"HCP: allergies first\n"
        f"Patient: {s['allergy']}\n"
        f"HCP: meds\n"
        f"Patient: {s['med']} daily\n"
        f"HCP: complaint\n"
        f"Patient: {s['complaint']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    if s["allergy"] != "none":
        soap["subjective"]["allergies"] = [_leaf(s["allergy"], s["allergy"])]
    return t, soap


def shape10(s):
    t = (
        f"Patient: {s['complaint']} doc\n"
        f"HCP: ok lets see {s['exam']}\n"
        f"HCP: probably {s['assessment']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["objective"]["exam"] = _leaf(s["exam"], s["exam"])
    soap["assessment"]["problems"] = [
        {"name": s["assessment"], "evidence_span": s["assessment"]}
    ]
    return t, soap


def shape11(s):
    t = (
        f"HCP: morning\n"
        f"Patient: morning um {s['complaint']} for {s['duration']}\n"
        f"HCP: any background\n"
        f"Patient: {s['pmh']}\n"
        f"HCP: plan {s['plan']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    hpi = f"{s['complaint']} for {s['duration']}"
    soap["subjective"]["hpi"] = _leaf(hpi, hpi)
    if s["pmh"] != "no significant history":
        soap["subjective"]["pmh"] = [_leaf(s["pmh"], s["pmh"])]
    soap["plan"]["items"] = [{"action": s["plan"], "evidence_span": s["plan"]}]
    return t, soap


def shape12(s):
    t = (
        f"Patient: been having {s['complaint']}\n"
        f"HCP: meds\n"
        f"Patient: {s['med']}\n"
        f"HCP: allergies\n"
        f"Patient: {s['allergy']}\n"
        f"HCP: exam {s['exam']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    if s["allergy"] != "none":
        soap["subjective"]["allergies"] = [_leaf(s["allergy"], s["allergy"])]
    soap["objective"]["exam"] = _leaf(s["exam"], s["exam"])
    return t, soap


def shape13(s):
    t = (
        f"HCP: whats the issue\n"
        f"Patient: like {s['complaint']} since {s['duration']}\n"
        f"HCP: history of {s['pmh']}\n"
        f"HCP: assessment {s['assessment']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    hpi = f"{s['complaint']} since {s['duration']}"
    soap["subjective"]["hpi"] = _leaf(hpi, hpi)
    if s["pmh"] != "no significant history":
        soap["subjective"]["pmh"] = [_leaf(s["pmh"], s["pmh"])]
    soap["assessment"]["problems"] = [
        {"name": s["assessment"], "evidence_span": s["assessment"]}
    ]
    return t, soap


def shape14(s):
    t = (
        f"HCP: meds and allergies\n"
        f"Patient: {s['med']} and {s['allergy']}\n"
        f"HCP: complaint\n"
        f"Patient: uh {s['complaint']}\n"
        f"HCP: plan {s['plan']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    if s["allergy"] != "none":
        soap["subjective"]["allergies"] = [_leaf(s["allergy"], s["allergy"])]
    soap["plan"]["items"] = [{"action": s["plan"], "evidence_span": s["plan"]}]
    return t, soap


def shape15(s):
    t = (
        f"Patient: ok so {s['complaint']} um {s['duration']}\n"
        f"HCP: pmh\n"
        f"Patient: {s['pmh']}\n"
        f"HCP: meds\n"
        f"Patient: {s['med']}\n"
        f"HCP: exam shows {s['exam']}\n"
        f"HCP: imp {s['assessment']} plan {s['plan']}"
    )
    soap = _empty_soap()
    soap["subjective"]["chief_complaint"] = _leaf(
        s["complaint"], s["complaint"]
    )
    hpi = f"{s['complaint']} um {s['duration']}"
    soap["subjective"]["hpi"] = _leaf(hpi, hpi)
    if s["pmh"] != "no significant history":
        soap["subjective"]["pmh"] = [_leaf(s["pmh"], s["pmh"])]
    soap["subjective"]["meds"] = [_leaf(s["med"], s["med"])]
    soap["objective"]["exam"] = _leaf(s["exam"], s["exam"])
    soap["assessment"]["problems"] = [
        {"name": s["assessment"], "evidence_span": s["assessment"]}
    ]
    soap["plan"]["items"] = [{"action": s["plan"], "evidence_span": s["plan"]}]
    return t, soap


SHAPES: list[Callable[[dict], tuple[str, dict]]] = [
    shape01,
    shape02,
    shape03,
    shape04,
    shape05,
    shape06,
    shape07,
    shape08,
    shape09,
    shape10,
    shape11,
    shape12,
    shape13,
    shape14,
    shape15,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _random_slots(rng: random.Random) -> dict:
    return {
        "complaint": rng.choice(COMPLAINTS),
        "duration": rng.choice(DURATIONS),
        "med": rng.choice(MEDS),
        "allergy": rng.choice(ALLERGIES),
        "pmh": rng.choice(PMH),
        "exam": rng.choice(EXAMS),
        "assessment": rng.choice(ASSESSMENTS),
        "plan": rng.choice(PLANS),
    }


def build_toy(
    n: int = 100, seed: int = 0, max_attempts: int = 10000
) -> list[dict]:
    """Build n unique (transcript, soap) pairs deterministically.

    Returns a list of dicts: {"transcript": str, "soap": dict}.

    Raises RuntimeError if n unique transcripts cannot be produced within
    max_attempts random samples.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[dict] = []

    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        shape = rng.choice(SHAPES)
        slots = _random_slots(rng)
        transcript, soap = shape(slots)
        if transcript in seen:
            continue
        seen.add(transcript)
        # Sanity: every populated evidence_span must be a substring of transcript.
        _assert_grounded(transcript, soap)
        out.append({"transcript": transcript, "soap": soap})

    if len(out) < n:
        raise RuntimeError(
            f"Could not produce {n} unique transcripts in {max_attempts} attempts; "
            f"got {len(out)}. Increase slot pool diversity."
        )
    return out


def _assert_grounded(transcript: str, soap: dict) -> None:
    """Walk the SOAP dict; every evidence_span must be a substring of transcript."""
    for span in _collect_spans(soap):
        if span is None:
            continue
        if span not in transcript:
            raise AssertionError(
                f"Generator bug: evidence_span {span!r} not in transcript:\n{transcript}"
            )


def _collect_spans(node) -> list:
    """Recursively collect all `evidence_span` values from a nested dict/list."""
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
