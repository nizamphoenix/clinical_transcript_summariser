"""
Core generation function: (template_name, spec) -> validated sample or None.

Wraps ollama.chat with format='json' and the validation pipeline. Retries
up to max_retries on validation failures, with a small temperature bump on
each retry to escape mode collapse.
"""

from __future__ import annotations

import json
from typing import Optional

from ollama import chat

from src.data_generation.templates import REGISTRY
from src.data_generation.validate import parse_and_validate

_USER_TEMPLATE = """\
Generate ONE {template_type} transcript matching these constraints:

- Length: {length} ({length_desc})
- Number of distinct issues / complaints: {issues}
- Conversational style: {style}
- Filler density: {fillers}
{extra}

Here is one worked example showing the required output shape and the \
verbatim-evidence-span discipline:

EXAMPLE OUTPUT:
{example}

Now produce ONE new {template_type} transcript as a JSON object with the \
same shape. Do not copy the example content — invent a fresh case. \
Output JSON only.
"""


def build_messages(template_name: str, spec: dict) -> list[dict]:
    """Assemble Ollama chat messages for a given template and stratification spec."""
    entry = REGISTRY[template_name]
    system = str(entry["system_prompt"]).format(schema=entry["spec"])

    # Build one-shot example with label nested under the right key
    one_shot = entry["one_shot"]
    example_obj = {
        "transcript": one_shot["transcript"],  # type: ignore[index]
        entry["label_key"]: one_shot["label"],  # type: ignore[index]
    }

    user = _USER_TEMPLATE.format(
        template_type=template_name.replace("_", " "),
        length=spec["length"],
        length_desc=spec["length_desc"],
        issues=spec.get("issues", 1),
        style=spec["style"],
        fillers=spec["fillers"],
        extra=spec.get("extra", ""),
        example=json.dumps(example_obj, indent=2),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_one(
    model: str,
    template_name: str,
    spec: dict,
    temperature: float = 0.7,
    max_retries: int = 2,
    num_predict: int = 2000,
) -> tuple[Optional[dict], str, list[str]]:
    """Generate one validated sample for the given template.

    Returns:
        (sample, status, attempts_log)
        sample: validated dict {"transcript": str, <label_key>: dict} or None.
        status: "ok" or last failure reason.
        attempts_log: list of one reason string per attempt.
    """
    messages = build_messages(template_name, spec)
    attempts_log: list[str] = []

    for attempt in range(max_retries + 1):
        temp = temperature + 0.05 * attempt
        try:
            resp = chat(
                model=model,
                messages=messages,
                format="json",
                # Qwen 3.x has a "thinking" capability — without think=False the
                # model burns the entire num_predict budget on a hidden reasoning
                # chain and emits no visible content. Disable it so output goes
                # straight to message.content. Ignored by non-thinking models.
                think=False,
                options={
                    "temperature": temp,
                    "num_predict": num_predict,
                    "num_ctx": 8192,
                },
            )
            raw = resp.message.content or ""
        except Exception as e:  # noqa: BLE001
            attempts_log.append(f"ollama_error: {type(e).__name__}: {e}")
            continue

        sample, reason = parse_and_validate(raw, template_name)
        attempts_log.append(reason)
        if sample is not None:
            sample["template"] = template_name
            return sample, "ok", attempts_log

    return (
        None,
        attempts_log[-1] if attempts_log else "no_attempts",
        attempts_log,
    )
