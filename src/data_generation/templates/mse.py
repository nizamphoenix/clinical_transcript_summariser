"""
MSE template — Psychiatric intake dialogue -> MSE JSON.

Used for zero-shot eval only (never seen during training).
Structurally distinct from SOAP and REFERRAL — tests true template-awareness.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Output spec
# ---------------------------------------------------------------------------

MSE_SPEC = """\
{
  "appearance":   {"text": <string|null>, "evidence_span": <string|null>},
  "behaviour":    {"text": <string|null>, "evidence_span": <string|null>},
  "speech":       {"text": <string|null>, "evidence_span": <string|null>},
  "mood":         {"text": <string|null>, "evidence_span": <string|null>},
  "affect":       {"text": <string|null>, "evidence_span": <string|null>},
  "thought":      {"text": <string|null>, "evidence_span": <string|null>},
  "cognition":    {"text": <string|null>, "evidence_span": <string|null>},
  "insight":      {"text": <string|null>, "evidence_span": <string|null>}
}

Rules:
- Every leaf with a text field must also have an evidence_span quoting the transcript verbatim (exact substring).
- Use null for fields not mentioned in the transcript.
- Output JSON only — no prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Data generation system prompt
# ---------------------------------------------------------------------------

MSE_SYSTEM_PROMPT = """\
You are a synthetic data generator producing realistic Australian psychiatric \
intake transcripts and their gold-standard Mental State Examination (MSE) \
JSON extraction.

You output a SINGLE JSON object with two top-level keys: "transcript" and \
"mse". No prose, no markdown fences, no explanation — just the JSON object.

HARD RULES (violations cause the sample to be discarded):

1. The transcript is a dialogue between a psychiatrist (HCP) and a patient. \
Each turn starts with "HCP:" or "Patient:" followed by a space.

2. Every evidence_span value in the mse object MUST be an EXACT VERBATIM \
SUBSTRING of the transcript text. Copy character-for-character. Use only \
straight ASCII apostrophes ('). Do not use curly/smart quotes. Do not \
paraphrase, do not change punctuation or capitalisation.

   WRONG: transcript says "can't concentrate", evidence_span is "can\u2019t concentrate"
   RIGHT: transcript says "can't concentrate", evidence_span is "can't concentrate"

3. The mse object follows this schema EXACTLY:

{schema}

4. Use null for any MSE domain not evidenced in the transcript. Never invent \
clinical observations the transcript does not support.

5. The "text" fields are short clinical summaries (may be paraphrased). \
The "evidence_span" fields are verbatim substrings from the transcript.

6. Realistic psychiatric intake language: patient may be guarded, distressed, \
or flat. HCP uses open questions, reflections, and direct probes. Australian \
context — public mental health setting, Medicare, "psych registrar".

MSE domain guidance:
  - appearance: how the patient looks (dress, grooming, eye contact)
  - behaviour: psychomotor activity (agitated, retarded, cooperative)
  - speech: rate, volume, tone (pressured, slow, monotone)
  - mood: patient's subjective report ("i feel...")
  - affect: clinician's observation of emotional expression (flat, labile, congruent)
  - thought: form and content (racing thoughts, rumination, delusions, SI)
  - cognition: orientation, memory, concentration
  - insight: patient's awareness of their condition
"""

# ---------------------------------------------------------------------------
# One-shot example
# ---------------------------------------------------------------------------

MSE_ONE_SHOT = {
    "transcript": (
        "HCP: hi, thanks for coming in today, how are you feeling\n"
        "Patient: not great, i haven't slept properly in weeks\n"
        "HCP: can you tell me more about that\n"
        "Patient: i just lie there, my mind won't stop racing, thinking \
about everything\n"
        "HCP: and how has your mood been\n"
        "Patient: low, really low, like there's no point to anything\n"
        "HCP: are you having any thoughts of harming yourself\n"
        "Patient: no nothing like that, i'm just exhausted\n"
        "HCP: you seem a little slowed down today, is that how you feel too\n"
        "Patient: yeah everything feels like it takes so much effort\n"
        "HCP: do you understand why you're here today\n"
        "Patient: i know i need help, i just don't know if anything will work"
    ),
    "label": {
        "appearance": {
            "text": None,
            "evidence_span": None,
        },
        "behaviour": {
            "text": "psychomotor slowing",
            "evidence_span": "you seem a little slowed down today",
        },
        "speech": {
            "text": None,
            "evidence_span": None,
        },
        "mood": {
            "text": "low mood, feels no point to anything",
            "evidence_span": "low, really low, like there's no point to anything",
        },
        "affect": {
            "text": None,
            "evidence_span": None,
        },
        "thought": {
            "text": "racing thoughts, no suicidal ideation",
            "evidence_span": "my mind won't stop racing, thinking about everything",
        },
        "cognition": {
            "text": "poor sleep, concentration difficulties",
            "evidence_span": "i haven't slept properly in weeks",
        },
        "insight": {
            "text": "partial insight — aware needs help but doubtful of treatment",
            "evidence_span": "i know i need help, i just don't know if anything will work",
        },
    },
}
