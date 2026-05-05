"""
REFERRAL template — GP referral letter -> REFERRAL JSON.

Two one-shot styles:
  - REFERRAL_A_ONE_SHOT: GP dictation monologue (used for training)
  - REFERRAL_B_ONE_SHOT: GP-patient dialogue about referral (used for zero-shot eval)

Both share the same REFERRAL_SPEC output schema — the difference is only
in the input transcript structure, making it a true template-awareness test.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output spec
# ---------------------------------------------------------------------------

REFERRAL_SPEC = """\
{
  "specialty":    {"text": <string|null>, "evidence_span": <string|null>},
  "patient":      {"text": <string|null>, "evidence_span": <string|null>},
  "reason":       {"text": <string|null>, "evidence_span": <string|null>},
  "history":      [{"text": <string>, "evidence_span": <string>}],
  "current_meds": [{"text": <string>, "evidence_span": <string>}],
  "request":      {"text": <string|null>, "evidence_span": <string|null>}
}

Rules:
- Every leaf with a text field must also have an evidence_span quoting the transcript verbatim (exact substring).
- Use null for scalar fields not mentioned in the transcript.
- Use [] for list fields with no items.
- Output JSON only — no prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Data generation system prompt (shared for A and B)
# ---------------------------------------------------------------------------

REFERRAL_SYSTEM_PROMPT = """\
You are a synthetic data generator producing realistic Australian GP referral \
transcripts and their gold-standard REFERRAL JSON extraction.

You output a SINGLE JSON object with two top-level keys: "transcript" and \
"referral". No prose, no markdown fences, no explanation — just the JSON object.

HARD RULES (violations cause the sample to be discarded):

1. The transcript format depends on the style requested:
   - Monologue (Style A): The HCP dictates the referral letter. Each line \
starts with "HCP:" followed by a space. No patient turns.
   - Dialogue (Style B): A GP-patient conversation where the GP explains the \
referral. Lines start with "HCP:" or "Patient:" followed by a space.

2. Every evidence_span value in the referral object MUST be an EXACT VERBATIM \
SUBSTRING of the transcript text. Copy character-for-character. Use only \
straight ASCII apostrophes ('). Do not use curly/smart quotes. Do not \
paraphrase, do not change punctuation or capitalisation.

   WRONG: transcript says "doesn't respond", evidence_span is "doesn\u2019t respond"
   RIGHT: transcript says "doesn't respond", evidence_span is "doesn't respond"

3. The referral object follows this schema EXACTLY:

{schema}

4. Use null for scalar fields not mentioned in the transcript. Use [] for list \
fields with no items. Never invent details the transcript does not contain.

5. The "text" fields are short clinical summaries (may be paraphrased). \
The "evidence_span" fields are verbatim substrings that justify the summary.

6. Australian clinical context: use "GP", "physio", "panadol", "ventolin", \
Medicare item numbers if relevant. Specialist names: Cardiology, Orthopaedics, \
Gastroenterology, Neurology, Respiratory, Rheumatology, Dermatology.
"""

# ---------------------------------------------------------------------------
# One-shot example A: GP dictation monologue (train)
# ---------------------------------------------------------------------------

REFERRAL_A_ONE_SHOT = {
    "transcript": (
        "HCP: referral to Cardiology\n"
        "HCP: patient is Margaret Liu, 62 years old\n"
        "HCP: she presents with exertional chest pain for the past 4 weeks\n"
        "HCP: background history of hypertension and type 2 diabetes\n"
        "HCP: current medications are metformin 500mg twice daily and \
perindopril 5mg daily\n"
        "HCP: no known drug allergies\n"
        "HCP: please review and advise on further cardiac workup"
    ),
    "label": {
        "specialty": {
            "text": "Cardiology",
            "evidence_span": "Cardiology",
        },
        "patient": {
            "text": "Margaret Liu, 62 years old",
            "evidence_span": "Margaret Liu, 62 years old",
        },
        "reason": {
            "text": "exertional chest pain for 4 weeks",
            "evidence_span": "exertional chest pain for the past 4 weeks",
        },
        "history": [
            {
                "text": "hypertension",
                "evidence_span": "hypertension",
            },
            {
                "text": "type 2 diabetes",
                "evidence_span": "type 2 diabetes",
            },
        ],
        "current_meds": [
            {
                "text": "metformin 500mg twice daily",
                "evidence_span": "metformin 500mg twice daily",
            },
            {
                "text": "perindopril 5mg daily",
                "evidence_span": "perindopril 5mg daily",
            },
        ],
        "request": {
            "text": "review and advise on cardiac workup",
            "evidence_span": "please review and advise on further cardiac workup",
        },
    },
}

# ---------------------------------------------------------------------------
# One-shot example B: GP-patient dialogue (zero-shot eval only)
# ---------------------------------------------------------------------------

REFERRAL_B_ONE_SHOT = {
    "transcript": (
        "HCP: so i think we need to refer you to a cardiologist\n"
        "Patient: oh right, is it serious\n"
        "HCP: i want to be careful, your chest pain on exertion is \
something we need to investigate properly\n"
        "Patient: ok how long have i had it, about 3 weeks i reckon\n"
        "HCP: yes and with your history of high blood pressure i don't \
want to miss anything\n"
        "Patient: i'm on ramipril for that\n"
        "HCP: correct, 5mg daily, i'll write to Cardiology and ask them \
to review you and arrange an exercise stress test"
    ),
    "label": {
        "specialty": {
            "text": "Cardiology",
            "evidence_span": "Cardiology",
        },
        "patient": {
            "text": None,
            "evidence_span": None,
        },
        "reason": {
            "text": "chest pain on exertion for 3 weeks",
            "evidence_span": "chest pain on exertion",
        },
        "history": [
            {
                "text": "high blood pressure",
                "evidence_span": "history of high blood pressure",
            }
        ],
        "current_meds": [
            {
                "text": "ramipril 5mg daily",
                "evidence_span": "ramipril for that",
            }
        ],
        "request": {
            "text": "review and arrange exercise stress test",
            "evidence_span": "review you and arrange an exercise stress test",
        },
    },
}
