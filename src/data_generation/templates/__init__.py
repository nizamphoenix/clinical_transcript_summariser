"""
Template registry.

Each template is a dict with keys:
  - spec          : str  — the JSON schema shown to the model at inference time
  - system_prompt : str  — generation system prompt (data gen only)
  - one_shot      : dict — {"transcript": str, "label": dict}
  - label_key     : str  — top-level key in the generated JSON label ("soap", "referral", "mse")
"""

from src.data_generation.templates.mse import (
    MSE_ONE_SHOT,
    MSE_SPEC,
    MSE_SYSTEM_PROMPT,
)
from src.data_generation.templates.referral import (
    REFERRAL_A_ONE_SHOT,
    REFERRAL_B_ONE_SHOT,
    REFERRAL_SPEC,
    REFERRAL_SYSTEM_PROMPT,
)
from src.data_generation.templates.soap import (
    SOAP_ONE_SHOT,
    SOAP_SPEC,
    SOAP_SYSTEM_PROMPT,
)

REGISTRY = {
    "soap": {
        "spec": SOAP_SPEC,
        "system_prompt": SOAP_SYSTEM_PROMPT,
        "one_shot": SOAP_ONE_SHOT,
        "label_key": "soap",
    },
    "referral_a": {
        "spec": REFERRAL_SPEC,
        "system_prompt": REFERRAL_SYSTEM_PROMPT,
        "one_shot": REFERRAL_A_ONE_SHOT,
        "label_key": "referral",
    },
    "referral_b": {
        "spec": REFERRAL_SPEC,
        "system_prompt": REFERRAL_SYSTEM_PROMPT,
        "one_shot": REFERRAL_B_ONE_SHOT,
        "label_key": "referral",
    },
    "mse": {
        "spec": MSE_SPEC,
        "system_prompt": MSE_SYSTEM_PROMPT,
        "one_shot": MSE_ONE_SHOT,
        "label_key": "mse",
    },
}

__all__ = [
    "REGISTRY",
    "SOAP_SPEC",
    "REFERRAL_SPEC",
    "MSE_SPEC",
]
