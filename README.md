# clinical_transcript_summariser

Template-aware clinical transcript extraction project.

The current focus is supervised fine-tuning a small instruction model to map raw
clinical transcripts into structured JSON across multiple output templates, with
evidence grounding via verbatim spans.

## Experiment Summary

| Experiment              | Train templates       | What was actually run                                                   | How to interpret it                                                                                                                              | Key caveat                                                                                                                  | Main artifacts                                                                                                                                                |
| ----------------------- | --------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v1` SOAP baseline      | `soap` only           | Held-out SOAP transcript-shape generalisation on a single output schema | Single-template baseline. If prompted with unseen schemas, it measures how far SOAP-only SFT plus prompt-time schema injection can transfer      | Not multi-template training. Success on an unseen schema would not by itself prove template-aware SFT                       | `notebooks/v1_soap_sft_baseline.ipynb`, `models/v1_qwen3b-soap-q4_k_m.gguf`                                                                                   |
| `v2` multi-template SFT | `soap` + `referral_a` | In-dist: `soap`, `referral_a`; zero-shot: `referral_b`, `mse`           | Multi-template baseline. Compare against `v1` to test whether multi-template SFT improves unseen-template transfer beyond schema injection alone | `referral_b` is unseen style with shared referral semantics; `mse` is the real unseen-ontology test and it failed zero-shot | `notebooks/v2-multitemplate-sft-report-README.md`, `notebooks/v2-multitemplate-sft-unsloth-200run-output.ipynb`, `models/v2_qwen3b-multitemplate-q4_k_m.gguf` |

## Current Takeaway

- `v1` is the comparison point for asking whether one-template training plus
  schema text is already enough.
- `v2` is the comparison point for asking whether multi-template training adds
  transfer beyond that baseline.
- Transfer to `referral_b` is encouraging, but it is a style-transfer result
  over the same referral schema as `referral_a`.
- Zero-shot `mse` shows schema text alone was not enough for transfer to a
  genuinely new clinical ontology.

## Project Structure

- `src/data_generation/`: synthetic data generation, validation, and template
  registry
- `src/data_generation/templates/`: template specs and one-shot examples for
  `soap`, `referral`, and `mse`
- `src/prompts.py`: template-aware inference prompt construction
- `scripts/gen_synth.py`: CLI for resumable synthetic data generation
- `notebooks/`: SFT experiments and saved Kaggle outputs

## v2 Report

The detailed write-up for the completed multi-template SFT run lives here:

- `notebooks/v2-multitemplate-sft-report-README.md`

That report covers:

- experiment context
- training and eval split design
- saved v2 metrics
- why `referral_b` transferred
- why zero-shot `mse` failed
- limitations of the current evaluation framing

## Quickstart

Install `mise`:

```bash
brew install mise
```

Activate `mise` in your shell:

```bash
mise activate zsh
```

Or add `eval "$(mise activate zsh)"` to your shell config.

Install pinned tools:

```bash
mise up
```

Sync Python versions:

```bash
make sync-py-versions
```

Set up the local Python environment with `uv`:

```bash
make setup-local-env
```

This will:

- create `.venv`
- install dependencies from `pyproject.toml`
- install pre-commit hooks

Run Python commands inside the environment with:

```bash
uv run python <script>.py
```

Useful Make targets:

- `make add-group-deps`
- `make remove-group-deps`

## Notes

- `data/` is gitignored. The current synthetic dataset used for v2 lives locally
  under `data/qwen3.5_latest/`.
- `notebooks/v2-multitemplate-sft-unsloth-200run-output.ipynb` is the saved
  Kaggle output artifact for the completed 200-step run.
- `notebooks/DEPRECATED_*.ipynb` files are kept only as older working notes and
  should not be treated as the current experiment record.
