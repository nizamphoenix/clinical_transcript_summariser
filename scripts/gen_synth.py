"""
CLI for synthetic clinical transcript generation via local Ollama models.

Usage:
    # Preview: 8 stratified samples (2 per template), one JSON per file.
    uv run python scripts/gen_synth.py --model qwen3.5:latest --preview

    # Bulk train: 50 SOAP + 50 REFERRAL-A mixed, streamed to JSONL.
    uv run python scripts/gen_synth.py --model qwen3.5:latest \\
        --template soap --n 50 --split train
    uv run python scripts/gen_synth.py --model qwen3.5:latest \\
        --template referral_a --n 50 --split train

    # Bulk eval (in-distribution):
    uv run python scripts/gen_synth.py --model qwen3.5:latest \\
        --template soap --n 20 --split eval_in_dist
    uv run python scripts/gen_synth.py --model qwen3.5:latest \\
        --template referral_a --n 20 --split eval_in_dist

    # Bulk eval (zero-shot held-out):
    uv run python scripts/gen_synth.py --model qwen3.5:latest \\
        --template referral_b --n 15 --split eval_zero_shot
    uv run python scripts/gen_synth.py --model qwen3.5:latest \\
        --template mse --n 15 --split eval_zero_shot

Output layout:
    data/{model_slug}/preview/
        01.json ... 08.json       # one valid sample per file
        _discards/NN.json         # failed attempts with reason
        _meta.json                # gen params + per-sample timings
    data/{model_slug}/{split}.{template}.jsonl
        # one valid sample per line, append mode (resumable)
    data/{model_slug}/{split}.{template}.discards.jsonl
        # one discard per line for prompt iteration
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from ollama import list as ollama_list

from src.data_generation.generate import generate_one
from src.data_generation.stratify import PREVIEW_SPECS, sample_bulk_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

VALID_TEMPLATES = ["soap", "referral_a", "referral_b", "mse"]


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def check_ollama(model: str) -> None:
    """Verify Ollama daemon is up and the requested model is installed."""
    try:
        resp = ollama_list()
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"ERROR: cannot reach Ollama daemon ({type(e).__name__}: {e}).\n"
            "Start it with: ollama serve  (in another terminal)"
        )

    available = []
    models_attr = getattr(resp, "models", None) or resp.get("models", [])
    for m in models_attr:
        tag = getattr(m, "model", None) or m.get("model") or m.get("name")
        if tag:
            available.append(tag)

    if model not in available:
        sys.exit(
            f"ERROR: model {model!r} not found in Ollama.\n"
            f"Available: {available}\n"
            f"Install with: ollama pull {model}"
        )


def run_preview(args: argparse.Namespace) -> None:
    out_dir = DATA_ROOT / model_slug(args.model) / "preview"
    discards_dir = out_dir / "_discards"
    out_dir.mkdir(parents=True, exist_ok=True)
    discards_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": args.model,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "samples": [],
    }

    print(f"Generating {len(PREVIEW_SPECS)} preview samples with {args.model}")
    for idx, spec in enumerate(PREVIEW_SPECS, start=1):
        template = spec["template"]
        t0 = time.perf_counter()
        sample, status, log = generate_one(
            model=args.model,
            template_name=template,
            spec=spec,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        elapsed = time.perf_counter() - t0

        spec_summary = (
            f"template={template} len={spec['length']} "
            f"issues={spec.get('issues', 1)} style={spec['style']}"
        )
        print(
            f"  [{idx}/{len(PREVIEW_SPECS)}] {spec_summary} "
            f"-> {status} ({elapsed:.1f}s, {len(log)} attempts)"
        )

        if sample is not None:
            path = out_dir / f"{idx:02d}.json"
            path.write_text(
                json.dumps(
                    {"spec": spec, **sample}, indent=2, ensure_ascii=False
                )
            )
        else:
            path = discards_dir / f"{idx:02d}.json"
            path.write_text(
                json.dumps(
                    {"spec": spec, "status": status, "attempts": log},
                    indent=2,
                    ensure_ascii=False,
                )
            )

        meta["samples"].append(
            {
                "idx": idx,
                "spec": spec,
                "status": status,
                "attempts": log,
                "elapsed_sec": round(elapsed, 2),
            }
        )

    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    n_ok = sum(1 for s in meta["samples"] if s["status"] == "ok")
    print(
        f"\nDone. {n_ok}/{len(PREVIEW_SPECS)} valid. "
        f"Output: {out_dir.relative_to(REPO_ROOT)}"
    )


def run_bulk(args: argparse.Namespace) -> None:
    out_dir = DATA_ROOT / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File per (split, template) so outputs don't collide when running in parallel
    out_path = out_dir / f"{args.split}.{args.template}.jsonl"
    discards_path = out_dir / f"{args.split}.{args.template}.discards.jsonl"

    # Resume support: count existing valid lines.
    existing = 0
    if out_path.exists():
        with out_path.open() as f:
            existing = sum(1 for _ in f)
        print(f"Resuming: {existing} samples already in {out_path.name}")

    rng = random.Random(args.seed)
    for _ in range(existing):
        sample_bulk_spec(rng, args.template)

    target = args.n
    n_ok = existing
    n_fail = 0
    t_start = time.perf_counter()

    print(
        f"Bulk generating: target={target} template={args.template} "
        f"model={args.model} seed={args.seed} split={args.split}"
    )

    try:
        with out_path.open("a") as fout, discards_path.open("a") as fdis:
            while n_ok < target:
                spec = sample_bulk_spec(rng, args.template)
                t0 = time.perf_counter()
                sample, status, log = generate_one(
                    model=args.model,
                    template_name=args.template,
                    spec=spec,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
                elapsed = time.perf_counter() - t0

                if sample is not None:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_ok += 1
                    rate = (n_ok - existing) / (
                        time.perf_counter() - t_start + 1e-9
                    )
                    eta_min = (target - n_ok) / rate / 60 if rate > 0 else 0
                    print(
                        f"  [{n_ok}/{target}] ok {elapsed:.1f}s "
                        f"(fail={n_fail}, eta={eta_min:.0f}min)"
                    )
                else:
                    fdis.write(
                        json.dumps(
                            {"spec": spec, "status": status, "attempts": log},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fdis.flush()
                    n_fail += 1
                    print(
                        f"  [{n_ok}/{target}] FAIL {status} ({elapsed:.1f}s, fail={n_fail})"
                    )

    except KeyboardInterrupt:
        print(f"\nInterrupted. {n_ok}/{target} valid in {out_path}")
        return

    print(
        f"\nDone. {n_ok}/{target} valid, {n_fail} discarded. "
        f"Output: {out_path.relative_to(REPO_ROOT)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="Ollama model tag, e.g. qwen3.5:latest",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Generate 8 stratified preview samples (2 per template).",
    )
    p.add_argument(
        "--template",
        choices=VALID_TEMPLATES,
        help="Template to generate (required for bulk mode).",
    )
    p.add_argument("--n", type=int, default=0, help="Bulk sample count.")
    p.add_argument(
        "--split",
        choices=["train", "eval_in_dist", "eval_zero_shot"],
        default="train",
        help="Dataset split — determines output filename.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    if not args.preview and args.n <= 0:
        p.error("must pass either --preview or --n N")
    if not args.preview and not args.template:
        p.error("--template is required for bulk mode")

    check_ollama(args.model)

    if args.preview:
        run_preview(args)
    else:
        run_bulk(args)


if __name__ == "__main__":
    main()
