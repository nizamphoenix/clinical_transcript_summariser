"""
Stratification specs for synthetic generation.

Two modes:
  - PREVIEW_SPECS: hand-crafted specs covering key corners of the design
    space, used for the preview-and-review phase.
  - sample_bulk_spec(rng, template): random sampler for bulk generation.

Axes:
  - template : soap | referral_a | referral_b | mse
  - length   : short | medium       (long removed — hurts validation pass rate)
  - issues   : 1 | 2 | 3            (number of distinct complaints/items)
  - style    : terse | conversational | warm
  - fillers  : low | medium | high

Dataset targets:
  - train       : soap (50) + referral_a (50) = 100 total
  - eval_in_dist: soap (20) + referral_a (20) = 40 total
  - eval_zeroshot: referral_b (15) + mse (15) = 30 total
"""

from __future__ import annotations

import random

_LENGTH_DESC = {
    "short": "5-8 turns total, focused single-issue visit",
    "medium": "10-15 turns total, includes history-taking and exam",
}

# No long — removed to improve validation pass rate and demo clarity.
_LENGTHS = ["short", "medium"]
_LENGTH_WEIGHTS = [0.70, 0.30]

_ISSUES = [1, 2, 3]
_ISSUE_WEIGHTS = [0.60, 0.30, 0.10]

_STYLES = ["terse", "conversational", "warm"]
_STYLE_WEIGHTS = [0.25, 0.55, 0.20]

_FILLERS = ["low", "medium", "high"]
_FILLER_WEIGHTS = [0.25, 0.50, 0.25]


# ---------------------------------------------------------------------------
# Preview specs — 8 hand-crafted samples (2 per template)
# ---------------------------------------------------------------------------

PREVIEW_SPECS: list[dict] = [
    # SOAP — short, single issue
    {
        "template": "soap",
        "length": "short",
        "length_desc": _LENGTH_DESC["short"],
        "issues": 1,
        "style": "terse",
        "fillers": "low",
    },
    # SOAP — medium, 2 issues
    {
        "template": "soap",
        "length": "medium",
        "length_desc": _LENGTH_DESC["medium"],
        "issues": 2,
        "style": "conversational",
        "fillers": "medium",
    },
    # REFERRAL A — short, monologue dictation
    {
        "template": "referral_a",
        "length": "short",
        "length_desc": _LENGTH_DESC["short"],
        "issues": 1,
        "style": "terse",
        "fillers": "low",
    },
    # REFERRAL A — medium, more history
    {
        "template": "referral_a",
        "length": "medium",
        "length_desc": _LENGTH_DESC["medium"],
        "issues": 2,
        "style": "conversational",
        "fillers": "low",
    },
    # REFERRAL B — short, dialogue (zero-shot eval style)
    {
        "template": "referral_b",
        "length": "short",
        "length_desc": _LENGTH_DESC["short"],
        "issues": 1,
        "style": "conversational",
        "fillers": "medium",
    },
    # REFERRAL B — medium, dialogue
    {
        "template": "referral_b",
        "length": "medium",
        "length_desc": _LENGTH_DESC["medium"],
        "issues": 1,
        "style": "warm",
        "fillers": "medium",
    },
    # MSE — short
    {
        "template": "mse",
        "length": "short",
        "length_desc": _LENGTH_DESC["short"],
        "issues": 1,
        "style": "conversational",
        "fillers": "low",
    },
    # MSE — medium
    {
        "template": "mse",
        "length": "medium",
        "length_desc": _LENGTH_DESC["medium"],
        "issues": 1,
        "style": "warm",
        "fillers": "medium",
    },
]


# ---------------------------------------------------------------------------
# Bulk spec sampler
# ---------------------------------------------------------------------------


def sample_bulk_spec(rng: random.Random, template: str) -> dict:
    """Random spec for bulk generation, weighted toward realistic mix.

    Args:
        rng: seeded Random instance for reproducibility.
        template: one of soap | referral_a | referral_b | mse.
    """
    length = rng.choices(_LENGTHS, weights=_LENGTH_WEIGHTS, k=1)[0]
    return {
        "template": template,
        "length": length,
        "length_desc": _LENGTH_DESC[length],
        "issues": rng.choices(_ISSUES, weights=_ISSUE_WEIGHTS, k=1)[0],
        "style": rng.choices(_STYLES, weights=_STYLE_WEIGHTS, k=1)[0],
        "fillers": rng.choices(_FILLERS, weights=_FILLER_WEIGHTS, k=1)[0],
    }
