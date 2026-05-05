"""
SOAP template — GP consultation dialogue -> SOAP JSON.

Used for:
  - Training data (50 samples)
  - In-distribution eval (20 samples, held-out stratification specs)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output spec (shown to model at inference time — same as src/prompts.py)
# ---------------------------------------------------------------------------

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
- Every leaf with a text/name/action field must also have an evidence_span quoting the transcript verbatim (exact substring).
- Use null for scalar fields not mentioned in the transcript.
- Use [] for list fields with no items.
- Output JSON only — no prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Data generation system prompt
# ---------------------------------------------------------------------------

SOAP_SYSTEM_PROMPT = """\
You are a synthetic data generator producing realistic Australian GP \
consultation transcripts and their gold-standard SOAP JSON extraction.

You output a SINGLE JSON object with two top-level keys: "transcript" and \
"soap". No prose, no markdown fences, no explanation — just the JSON object.

HARD RULES (violations cause the sample to be discarded):

1. The transcript is plain text with turns separated by newlines. Each turn \
starts with exactly "HCP:" or "Patient:" followed by a space. Turns generally \
alternate but the HCP may take consecutive turns when examining or explaining.

2. Every evidence_span value in the soap object MUST be an EXACT VERBATIM \
SUBSTRING of the transcript text. Copy character-for-character from the \
transcript. Use only straight ASCII apostrophes ('). Do not use curly/smart \
quotes. Do not paraphrase, do not change punctuation, do not change \
capitalisation.

   WRONG: transcript says "doesn't look infected", evidence_span is "doesn\u2019t look infected"
   RIGHT: transcript says "doesn't look infected", evidence_span is "doesn't look infected"

3. The soap object follows this schema EXACTLY:

{schema}

4. Use null for scalar fields (chief_complaint.text, hpi.text, exam.text, \
and their evidence_spans) when the transcript does not mention them. \
Use [] for list fields (pmh, meds, allergies, problems, items) when nothing \
applies. Never invent details that the transcript does not contain.

5. The "text"/"name"/"action" fields are short clinical summaries (may be \
paraphrased). The "evidence_span" fields are verbatim substrings of the \
transcript that justify the summary.

6. Produce realistic GP-style language: fillers ("uh", "um", "like", "you \
know"), false starts, colloquial speech from the patient, more clinical \
phrasing from the HCP. Australian context (panadol, ventolin, "BP 140 over \
90", "GP", "physio").
"""

# ---------------------------------------------------------------------------
# One-shot example
# ---------------------------------------------------------------------------

SOAP_ONE_SHOT = {
    "transcript": (
        "HCP: morning, what brings you in today\n"
        "Patient: uh i've had this sore throat for like 3 days now\n"
        "HCP: any fever\n"
        "Patient: bit warm last night yeah\n"
        "HCP: any meds you take\n"
        "Patient: just panadol when needed\n"
        "HCP: allergies\n"
        "Patient: penicillin\n"
        "HCP: ok let me have a look, throat looks red and inflamed\n"
        "HCP: looks like a viral pharyngitis, rest, fluids, "
        "panadol as needed, come back if it gets worse"
    ),
    "label": {
        "subjective": {
            "chief_complaint": {
                "text": "sore throat",
                "evidence_span": "sore throat",
            },
            "hpi": {
                "text": "sore throat for 3 days, fever last night",
                "evidence_span": "sore throat for like 3 days now",
            },
            "pmh": [],
            "meds": [
                {
                    "text": "panadol as needed",
                    "evidence_span": "just panadol when needed",
                }
            ],
            "allergies": [
                {"text": "penicillin", "evidence_span": "penicillin"}
            ],
        },
        "objective": {
            "exam": {
                "text": "throat red and inflamed",
                "evidence_span": "throat looks red and inflamed",
            }
        },
        "assessment": {
            "problems": [
                {
                    "name": "viral pharyngitis",
                    "evidence_span": "viral pharyngitis",
                }
            ]
        },
        "plan": {
            "items": [
                {"action": "rest and fluids", "evidence_span": "rest, fluids"},
                {
                    "action": "panadol as needed",
                    "evidence_span": "panadol as needed",
                },
                {
                    "action": "return if worse",
                    "evidence_span": "come back if it gets worse",
                },
            ]
        },
    },
}
