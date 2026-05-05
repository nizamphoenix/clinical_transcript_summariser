"""
One-off migration: inject a "template" field into every record in existing
JSONL files under data/qwen3.5_latest/.

Template is inferred from the filename stem, e.g.:
  train.soap.jsonl          -> "soap"
  train.referral_a.jsonl    -> "referral_a"
  eval_in_dist.referral_a.jsonl -> "referral_a"
  eval_zero_shot.referral_b.jsonl -> "referral_b"
  eval_zero_shot.mse.jsonl  -> "mse"

Skips files that already have "template" on every record.
Skips discard files (*discards*).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "qwen3.5_latest"

KNOWN_TEMPLATES = ["soap", "referral_a", "referral_b", "mse"]


def infer_template(stem: str) -> str | None:
    """Extract template name from filename stem (longest match wins)."""
    for t in sorted(KNOWN_TEMPLATES, key=len, reverse=True):
        if t in stem:
            return t
    return None


def migrate_file(path: Path) -> None:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    if not records:
        print(f"  SKIP (empty): {path.name}")
        return

    if all("template" in r for r in records):
        print(f"  SKIP (already migrated): {path.name}")
        return

    template = infer_template(path.stem)
    if template is None:
        print(f"  WARN (cannot infer template): {path.name}")
        return

    migrated = []
    for r in records:
        if "template" not in r:
            r["template"] = template
        migrated.append(r)

    path.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in migrated)
        + "\n"
    )
    print(f"  OK ({len(migrated)} records, template={template!r}): {path.name}")


def main() -> None:
    jsonl_files = sorted(
        p for p in DATA_DIR.glob("*.jsonl") if "discards" not in p.name
    )

    if not jsonl_files:
        print(f"No JSONL files found in {DATA_DIR}")
        return

    print(
        f"Migrating {len(jsonl_files)} JSONL files in {DATA_DIR.relative_to(REPO_ROOT)}\n"
    )
    for path in jsonl_files:
        migrate_file(path)
    print("\nDone.")


if __name__ == "__main__":
    main()
