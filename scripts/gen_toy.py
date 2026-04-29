"""CLI to materialise toy training data.

Writes data/toy.train.jsonl and data/toy.eval.jsonl. The data/ directory is
gitignored — this script is the source of truth for reproducing it.

v1 mode (no holdout, train/eval drawn from the same shapes):
    python scripts/gen_toy.py --n-train 90 --n-eval 10 --held-out 0 --seed 0

v1.1 mode (held-out shapes — true generalisation test):
    python scripts/gen_toy.py --n-train 500 --n-eval 50 \
        --held-out 2 --holdout-seed 42 --seed 0

Train uses (15 - held_out) shapes; eval uses ONLY the held-out shapes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.toy_data import SHAPES, build_toy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-train", type=int, default=90, help="Number of train examples."
    )
    parser.add_argument(
        "--n-eval", type=int, default=10, help="Number of eval examples."
    )
    parser.add_argument(
        "--held-out",
        type=int,
        default=0,
        help="How many shapes to hold out for eval. "
        "0 = train and eval share all shapes (v1 mode).",
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=42,
        help="Seed for choosing which shapes to hold out.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Data RNG seed (eval uses seed+1)."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_shapes = len(SHAPES)
    if args.held_out < 0 or args.held_out >= n_shapes:
        parser.error(f"--held-out must be in [0, {n_shapes - 1}]")

    all_indices = list(range(n_shapes))
    if args.held_out == 0:
        train_shapes = all_indices
        eval_shapes = all_indices
    else:
        held = sorted(
            random.Random(args.holdout_seed).sample(all_indices, args.held_out)
        )
        train_shapes = [i for i in all_indices if i not in held]
        eval_shapes = held

    print(f"shapes total:        {n_shapes}")
    print(f"train shape indices: {train_shapes}")
    print(
        f"eval  shape indices: {eval_shapes}"
        + ("  (held-out)" if args.held_out > 0 else "  (shared with train)")
    )

    train = build_toy(
        n=args.n_train,
        seed=args.seed,
        allowed_shape_indices=train_shapes,
    )
    eval_ = build_toy(
        n=args.n_eval,
        seed=args.seed + 1,
        allowed_shape_indices=eval_shapes,
    )

    train_path = args.out_dir / "toy.train.jsonl"
    eval_path = args.out_dir / "toy.eval.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(eval_path, eval_)

    print(f"wrote {len(train)} train examples -> {train_path}")
    print(f"wrote {len(eval_)} eval  examples -> {eval_path}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
