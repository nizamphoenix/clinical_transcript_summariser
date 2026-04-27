"""Local smoke test: run the same 4-metric eval as Kaggle cell 12 against a GGUF
served via llama-cpp-python. Confirms the merge -> convert -> quantise pipeline
preserved model quality.

Usage:
    uv run python scripts/smoke_local.py --model models/qwen3b-soap-q4_k_m.gguf
    uv run python scripts/smoke_local.py --model models/qwen3b-soap-f16.gguf --show 3
    uv run python scripts/smoke_local.py --model models/qwen3b-soap-q4_k_m.gguf \
        --out outputs/smoke_q4.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from llama_cpp import Llama

from src.prompts import build_messages
from src.toy_data import _collect_spans

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = REPO_ROOT / "data" / "toy.eval.jsonl"
GEN_TOY = REPO_ROOT / "scripts" / "gen_toy.py"


def ensure_eval_data(path: Path) -> None:
    """Regenerate toy data with the canonical seed if the eval file is missing."""
    if path.exists():
        return
    print(f"[setup] {path} missing -> regenerating via gen_toy.py")
    subprocess.run(
        [
            sys.executable,
            str(GEN_TOY),
            "--n",
            "100",
            "--seed",
            "0",
            "--out-dir",
            str(path.parent),
        ],
        check=True,
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def precheck(model_path: Path, eval_path: Path) -> list[dict]:
    if not model_path.exists():
        sys.exit(f"[error] model not found: {model_path}")
    size_gb = model_path.stat().st_size / 1e9
    if size_gb < 1.0:
        sys.exit(f"[error] model file suspiciously small: {size_gb:.2f} GB")
    print(f"[ok] model {model_path.name} ({size_gb:.2f} GB)")

    ensure_eval_data(eval_path)
    rows = load_jsonl(eval_path)
    if not rows:
        sys.exit(f"[error] eval file is empty: {eval_path}")
    print(f"[ok] eval {eval_path.name} ({len(rows)} rows)")
    return rows


def make_generate(llm: Llama, max_tokens: int):
    def generate(transcript: str) -> str:
        msgs = build_messages(transcript)
        resp = llm.create_chat_completion(
            messages=msgs,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()

    return generate


def score(pred_str: str, gold: dict, transcript: str) -> dict:
    # (1) JSON parse
    try:
        pred = json.loads(pred_str)
    except Exception:
        return {"parse": 0, "keys": 0, "cc": 0.0, "ground": 0.0}
    # (2) schema keys
    keys = int(
        set(pred.keys()) >= {"subjective", "objective", "assessment", "plan"}
    )
    # (3) chief_complaint Jaccard
    try:
        p_cc = (pred["subjective"]["chief_complaint"]["text"] or "").lower()
        g_cc = (gold["subjective"]["chief_complaint"]["text"] or "").lower()
        a, b = set(p_cc.split()), set(g_cc.split())
        cc = len(a & b) / max(len(a | b), 1)
    except Exception:
        cc = 0.0
    # (4) evidence grounding
    spans = [s for s in _collect_spans(pred) if s]
    if not spans:
        ground = 0.0
    else:
        ground = sum(1 for s in spans if s in transcript) / len(spans)
    return {"parse": 1, "keys": keys, "cc": cc, "ground": ground}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, type=Path, help="Path to .gguf model file"
    )
    ap.add_argument(
        "--eval",
        default=DEFAULT_EVAL,
        type=Path,
        help="Path to eval JSONL (default: data/toy.eval.jsonl)",
    )
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="-1 = offload all layers to Metal/GPU; 0 = CPU only",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=2,
        help="Number of qualitative TRANSCRIPT/PRED/GOLD blocks to print",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSONL path to dump per-example results",
    )
    args = ap.parse_args()

    rows = precheck(args.model, args.eval)

    print(
        f"[load] llama_cpp.Llama(n_ctx={args.n_ctx}, n_gpu_layers={args.n_gpu_layers})"
    )
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
        seed=0,
    )
    print(f"[load] ready in {time.perf_counter() - t0:.1f}s")

    generate = make_generate(llm, args.max_tokens)

    results: list[dict] = []
    preds: list[str] = []
    latencies: list[float] = []

    print(f"[run] generating on {len(rows)} examples ...")
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        pred = generate(row["transcript"])
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        preds.append(pred)
        scores = score(pred, row["soap"], row["transcript"])
        results.append(scores)
        print(
            f"  [{i + 1:>2}/{len(rows)}] {dt_ms:>6.0f} ms  "
            f"parse={scores['parse']} keys={scores['keys']} "
            f"cc={scores['cc']:.2f} ground={scores['ground']:.2f}"
        )

    n = len(results)
    summary = {
        "n": n,
        "json_parse_rate": sum(r["parse"] for r in results) / n,
        "schema_keys_rate": sum(r["keys"] for r in results) / n,
        "cc_overlap_mean": sum(r["cc"] for r in results) / n,
        "evidence_grounding_rate": sum(r["ground"] for r in results) / n,
        "mean_latency_ms": sum(latencies) / n,
        "model": args.model.name,
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for row, pred, scores, dt in zip(rows, preds, results, latencies):
                f.write(
                    json.dumps(
                        {
                            "transcript": row["transcript"],
                            "gold": row["soap"],
                            "pred_raw": pred,
                            "scores": scores,
                            "latency_ms": dt,
                        }
                    )
                    + "\n"
                )
            f.write(json.dumps({"_summary": summary}) + "\n")
        print(f"[out] wrote per-example results -> {args.out}")

    for i in range(min(args.show, len(rows))):
        print("=" * 72)
        print("TRANSCRIPT:")
        print(rows[i]["transcript"])
        print("\nPREDICTED:")
        print(preds[i])
        print("\nGOLD:")
        print(json.dumps(rows[i]["soap"], indent=2))


if __name__ == "__main__":
    main()
