"""Gradio demo for the clinical transcript summariser.

Standalone — loads the GGUF model directly via llama-cpp-python (no FastAPI yet).
Once src/server.py exists, swap `_extract_local` for an HTTP call.

Usage:
    uv run python -m src.demo
    MODEL_PATH=models/qwen3b-soap-q4_k_m_30steps.gguf uv run python -m src.demo
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
from llama_cpp import Llama

from src.prompts import build_messages

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "qwen3b-soap-q4_k_m_30steps.gguf"
EVAL_PATH = REPO_ROOT / "data" / "toy.eval.jsonl"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL)))
N_CTX = int(os.environ.get("N_CTX", "4096"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = all on Metal


@lru_cache(maxsize=1)
def _llm() -> Llama:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(
        f"[demo] loading {MODEL_PATH.name} (n_ctx={N_CTX}, n_gpu_layers={N_GPU_LAYERS})"
    )
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
        seed=0,
    )


def _extract_local(transcript: str) -> str:
    """Call the local GGUF model. Returns raw string from the model."""
    llm = _llm()
    resp = llm.create_chat_completion(  # type: ignore[arg-type]
        messages=build_messages(transcript),
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    return resp["choices"][0]["message"]["content"].strip()  # type: ignore[index,union-attr]


def extract(transcript: str) -> tuple[dict | str, str]:
    """Gradio handler. Returns (parsed_json_or_error, raw_string)."""
    if not transcript or not transcript.strip():
        return {"error": "Please enter a transcript."}, ""
    try:
        raw = _extract_local(transcript)
    except FileNotFoundError as e:
        return {"error": str(e)}, ""
    except Exception as e:
        return {"error": f"Inference failed: {type(e).__name__}: {e}"}, ""
    try:
        parsed = json.loads(raw)
        return parsed, raw
    except json.JSONDecodeError as e:
        return {"error": f"Model returned invalid JSON: {e}"}, raw


def _load_examples(n: int = 3) -> list[list[str]]:
    """Pull a few transcripts from toy.eval.jsonl for the Examples panel."""
    if not EVAL_PATH.exists():
        return []
    out: list[list[str]] = []
    with EVAL_PATH.open() as f:
        for line in f:
            if len(out) >= n:
                break
            row = json.loads(line)
            out.append([row["transcript"]])
    return out


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Clinical Transcript Summariser") as app:
        gr.Markdown(
            "# Clinical Transcript Summariser\n"
            "Paste a short clinical transcript. The model extracts a SOAP-shaped "
            "JSON with evidence spans grounded in the transcript.\n\n"
            f"**Model:** `{MODEL_PATH.name}` (Q4_K_M, runs locally via llama.cpp)"
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(
                    label="Transcript",
                    placeholder="HCP: what brings you in?\nPatient: chest pain for 2 days\n...",
                    lines=12,
                    max_lines=30,
                )
                btn = gr.Button("Extract SOAP", variant="primary")
                gr.Examples(
                    examples=_load_examples(), inputs=inp, label="Examples"
                )
            with gr.Column(scale=1):
                out_json = gr.JSON(label="Parsed SOAP")
                out_raw = gr.Textbox(
                    label="Raw model output", lines=8, max_lines=20
                )

        btn.click(fn=extract, inputs=inp, outputs=[out_json, out_raw])
    return app


if __name__ == "__main__":
    build_app().launch(server_port=int(os.environ.get("PORT", "7860")))
