"""CLI to materialise toy training data.

Writes data/toy.train.jsonl and data/toy.eval.jsonl. The data/ directory is
gitignored — this script is the source of truth for reproducing it.

Usage:
    python scripts/gen_toy.py --n 100 --seed 0 --out-dir data/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.toy_data import build_toy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=100, help="Total examples (train + eval)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for determinism."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data"), help="Output directory."
    )
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=0.1,
        help="Fraction held out for eval.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_toy(n=args.n, seed=args.seed)
    n_eval = max(1, int(round(args.n * args.eval_frac)))
    train, eval_ = pairs[:-n_eval], pairs[-n_eval:]

    train_path = args.out_dir / "toy.train.jsonl"
    eval_path = args.out_dir / "toy.eval.jsonl"

    _write_jsonl(train_path, train)
    _write_jsonl(eval_path, eval_)

    print(f"Wrote {len(train)} train examples to {train_path}")
    print(f"Wrote {len(eval_)} eval  examples to {eval_path}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
