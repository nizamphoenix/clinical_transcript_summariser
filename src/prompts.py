"""
Prompts and output specs for the clinical transcript summariser.

NOTE: SOAP_SPEC and related constants now live in
src/data_generation/templates/soap.py. This module re-exports them for
backward compatibility with the Kaggle notebook and any other existing code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backward-compat re-exports
# ---------------------------------------------------------------------------
from src.data_generation.templates import REGISTRY  # noqa: F401
from src.data_generation.templates.soap import SOAP_SPEC  # noqa: F401

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


def build_inference_messages(template_name: str, transcript: str) -> list[dict]:
    """Assemble chat messages for inference given a template name.

    Looks up the output spec from REGISTRY so the model knows which JSON
    shape to produce. Used by the v2+ Kaggle notebook and the FastAPI server.
    """
    spec = REGISTRY[template_name]["spec"]
    return build_messages(transcript, template_spec=str(spec))


def build_messages(
    transcript: str, template_spec: str | None = None
) -> list[dict]:
    """Assemble chat messages for the model.

    Used by the Kaggle training notebook and smoke_local.py.
    v1 always passes SOAP_SPEC. v2+ passes one of {SOAP_SPEC, REFERRAL_SPEC, MSE_SPEC}.
    """
    if template_spec is None:
        template_spec = SOAP_SPEC
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                template_spec=template_spec, transcript=transcript
            ),
        },
    ]
