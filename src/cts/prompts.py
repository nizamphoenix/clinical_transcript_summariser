"""
Prompts and output specs for the clinical transcript summariser.

================================================================================
Two senses of "template" — read this first
================================================================================

The codebase uses the word "template" in two different ways. They are unrelated
and must not be confused.

1. OUTPUT TEMPLATE / TEMPLATE SPEC
   - Lives in this module (e.g. SOAP_SPEC).
   - A textual description of the JSON schema the model must produce.
   - Passed to the model in every prompt.
   - In v1 there is exactly one (SOAP). v2 will add referral letter, MSE, etc.
   - This is the "template" Heidi clinicians author in production.

2. TRANSCRIPT-SHAPE TEMPLATE
   - Lives in src/cts/toy_data.py.
   - A Python f-string used internally by the toy data generator to fabricate
     fake transcripts (different turn orders, filler densities, etc.).
   - Never seen by the model.
   - In v1 there are 15 of these for linguistic variety in the toy data.

================================================================================
Architectural decision: template-aware extraction (Pattern B, "Option C")
================================================================================

WHY this approach
-----------------
Heidi's product lets clinicians author their own note formats. At inference time
the model receives (template_spec, transcript) and must produce JSON conforming
to the spec. Two ways to build this:

  (A) One model per template. Doesn't scale — Heidi has hundreds of templates.
  (B) One template-aware model that reads the spec from its input.

We pick (B) — same posture as Heidi (clinicians author templates; no retraining
per template).

WHAT v1 actually proves
-----------------------
v1 trains on 100 examples that ALL use SOAP_SPEC. The prompt format is already
template-aware (template_spec is a slot in USER_TEMPLATE), but because every
training example uses the same spec, the model has no incentive to actually
*read* the spec. It could equally well learn "ignore the spec, always emit
SOAP" and score identically on SOAP-only eval.

Therefore v1 demonstrates ONLY:
  - the training pipeline works (data → SFT → save → infer → eval)
  - the model can produce well-formed SOAP JSON with grounded evidence spans

v1 does NOT demonstrate template-awareness. That requires multiple specs in
training and a held-out spec at eval — deferred to v2.

WHY pass the spec in v1 anyway
------------------------------
Two reasons:
  1. Interface stability — v2 adds more specs as a pure data change, no code
     refactor.
  2. The pre-trained Qwen base already has weak instruction-following; passing
     the spec preserves any latent template-awareness rather than training it
     out.

TRADE-OFFS we accepted
----------------------
  - Spike doesn't differentiate (a) "learned template-awareness" from (b) "memorised
    SOAP". README must be honest about this.
  - With only 100 examples and 20 steps, evidence-span learning is the hardest
    objective. We add an explicit grounding metric to track it.
"""

# ---------------------------------------------------------------------------
# SOAP output spec
# ---------------------------------------------------------------------------
# Shown to the model as a textual description of the JSON it must produce.
# Every leaf field has an `evidence_span` for hallucination grounding.
# `vitals` intentionally omitted from v1 — toy transcripts won't contain them.

SOAP_SPEC = """\
{
  "subjective": {
    "chief_complaint": {"text": <string|null>, "evidence_span": <string|null>},
    "hpi":             {"text": <string|null>, "evidence_span": <string|null>},
    "pmh":             [{"text": <string>, "evidence_span": <string>}],
    "meds":            [{"text": <string>, "evidence_span": <string>}],
    "allergies":       [{"text": <string>, "evidence_span": <string>}]
  },
  "objective": {
    "exam": {"text": <string|null>, "evidence_span": <string|null>}
  },
  "assessment": {
    "problems": [{"name": <string>, "evidence_span": <string>}]
  },
  "plan": {
    "items": [{"action": <string>, "evidence_span": <string>}]
  }
}

Rules:
- Every leaf with a `text`/`name`/`action` field must also have an
  `evidence_span` quoting the transcript verbatim (exact substring).
- Use null for scalar fields not mentioned in the transcript.
- Use [] for list fields with no items.
- Output JSON only — no prose, no markdown fences.
"""


SYSTEM_PROMPT = (
    "You are a clinical scribe. Given a TEMPLATE SPEC and a TRANSCRIPT, "
    "extract a JSON object that conforms exactly to the spec. "
    "Every leaf field must include an `evidence_span` quoting the transcript "
    "verbatim. Use null for fields not mentioned. Output JSON only — no prose."
)


USER_TEMPLATE = (
    "TEMPLATE SPEC:\n{template_spec}\n\n"
    "TRANSCRIPT:\n{transcript}\n\n"
    "Return JSON matching the spec."
)


def build_messages(
    transcript: str, template_spec: str = SOAP_SPEC
) -> list[dict]:
    """Assemble chat messages for the model.

    v1 always passes SOAP_SPEC. v2 will pass one of {SOAP_SPEC, REFERRAL_SPEC, ...}.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                template_spec=template_spec, transcript=transcript
            ),
        },
    ]
