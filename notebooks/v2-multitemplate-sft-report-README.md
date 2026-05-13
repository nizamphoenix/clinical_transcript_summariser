# v2-multitemplate-sft Report

## Summary

`v2-multitemplate-sft` tested whether a small SFT model could learn
template-aware clinical extraction when the output schema is injected at
inference time.

The run succeeded on:

- `soap` in-distribution
- `referral_a` in-distribution
- `referral_b` zero-shot transcript-style transfer

The run failed on:

- `mse` zero-shot field-vocabulary transfer

The important result is not that SFT failed outright. It learned
schema-following and evidence-grounded extraction for seen template families,
but it did not learn to map psychiatric language into a genuinely unseen output
ontology from schema text alone.

## Experiment Context

### Task framing

The project uses template-aware prompting rather than a single fixed output
format.

At inference time the model is shown:

- a global system instruction from `src/prompts.py`
- a template spec pulled from `src.data_generation.templates.REGISTRY`
- the raw transcript

The inference path is implemented in `src/prompts.py` via
`build_inference_messages(template_name, transcript)`.

### Training templates

The v2 training mix contained only:

- `soap`
- `referral_a`

The held-out zero-shot templates were:

- `referral_b`
- `mse`

This matters because the two held-out tasks were not equally novel:

- `referral_b` shares the same output schema as `referral_a`; only the
  transcript style changes
- `mse` introduces new field semantics: `appearance`, `behaviour`, `speech`,
  `mood`, `affect`, `thought`, `cognition`, `insight`

So `referral_b` mostly tested transfer across transcript phrasing, while `mse`
tested transfer to a new clinical extraction ontology.

### Data

The synthetic dataset contained 140 grounded, schema-valid examples:

- train: 100
  - 50 `soap`
  - 50 `referral_a`
- eval in-distribution: 20
  - 10 `soap`
  - 10 `referral_a`
- eval zero-shot: 20
  - 10 `referral_b`
  - 10 `mse`

Data generation and template specs live under `src/data_generation/`.

### Training setup

The original `bitsandbytes + peft` notebook hit Kaggle T4 VRAM limits.

The completed run used:

- Unsloth
- Kaggle T4
- `use_gradient_checkpointing='unsloth'`
- `MAX_STEPS=200`

Saved artifacts:

- canonical notebook: `notebooks/v2_multitemplate_sft_unsloth.ipynb`
- saved run output: `notebooks/v2-multitemplate-sft-unsloth-200run-output.ipynb`

## Prompting Context

The global extraction prompt in `src/prompts.py` tells the model to:

- conform exactly to the provided template spec
- include verbatim `evidence_span` values
- use `null` for fields not mentioned
- output JSON only

That `use null for fields not mentioned` instruction was useful for
hallucination control, but it also created a safe failure mode: when the model
was unsure how to map evidence into an unseen schema, it could remain compliant
by returning `null`.

## Results

### Success criteria

- `json_parse_rate >= 0.8`
- `schema_keys_rate >= 0.7`
- `key_overlap_mean >= 0.4`
- `evidence_grounding_rate >= 0.5`

### Saved metrics

| Template     | Split     | json_parse_rate | schema_keys_rate | key_overlap_mean | evidence_grounding_rate |
| ------------ | --------- | --------------: | ---------------: | ---------------: | ----------------------: |
| `soap`       | in-dist   |            1.00 |             1.00 |             0.83 |                    0.96 |
| `referral_a` | in-dist   |            1.00 |             1.00 |             0.74 |                    0.97 |
| `referral_b` | zero-shot |            1.00 |             1.00 |             0.42 |                    0.99 |
| `mse`        | zero-shot |            1.00 |             1.00 |             0.00 |                    0.15 |

Overall:

- in-distribution performance was strong
- zero-shot performance was split between one easy transfer case (`referral_b`)
  and one real failure case (`mse`)

## Error Analysis: Why v2 SFT Failed On MSE

### Observed failure mode

The model did not fail by producing malformed JSON.

It failed by producing schema-shaped but clinically empty output.

An example `mse` prediction from the saved run output was:

```json
{
  "appearance": null,
  "behaviour": null,
  "speech": null,
  "mood": null,
  "affect": null,
  "thought": null,
  "cognition": null,
  "insight": null
}
```

That prediction came from a transcript containing obvious evidence for several
MSE domains, including anxiety, poor concentration, fidgeting, fast speech,
paranoid thoughts, and partial insight.

This was not a lack-of-evidence case. It was a field-mapping failure.

### Root cause

#### 1. No supervision for the MSE ontology

The model was never trained on examples mapping transcript evidence into MSE
fields.

SFT taught it how to do:

- SOAP extraction
- referral extraction
- evidence-span grounding
- schema-shaped JSON output

It did not teach:

- that `feeling really anxious` should populate `mood`
- that `talking fast and fidgeting` should populate `speech` or `behaviour`
- that `people watching me` should populate `thought`
- that `i know i need help` should populate `insight`

The schema text alone was not enough to induce that mapping reliably.

#### 2. The null policy gave the model a safe fallback

The inference prompt says to use `null` for fields not mentioned.

The `mse` spec repeats the same conservative instruction.

Under uncertainty, the model had two choices:

- attempt a semantic mapping into unfamiliar field names
- stay schema-compliant and avoid hallucination by null-filling the template

It chose the second path.

So the failure mode was not hallucination. It was over-conservative abstention.

#### 3. `referral_b` was easier than it looked

`referral_b` transferred because it reused the same output schema as
`referral_a`.

That means v2 already knew the label semantics for:

- `specialty`
- `patient`
- `reason`
- `history`
- `current_meds`
- `request`

The remaining challenge was mostly handling a different transcript shape.

That is useful, but it is not the same as generalizing to an unseen schema
vocabulary.

#### 4. Small-data SFT learned formatting more easily than ontology transfer

With 100 training examples across two template families, the model learned:

- JSON compliance
- top-level schema structure
- evidence-span behavior

That amount of data was enough for formatting discipline, but not enough to
consistently learn cross-template semantic abstraction over a new clinical
domain.

## Metric Caveats

Two caveats matter when interpreting the current v2 numbers.

### `schema_keys_rate` is shallow

The scoring code checks whether the required top-level keys are present.

It does not fully validate the nested shape for each field. For MSE, a
prediction like:

```json
{
  "appearance": null,
  "behaviour": null,
  "speech": null,
  "mood": null,
  "affect": null,
  "thought": null,
  "cognition": null,
  "insight": null
}
```

still counts as schema-key correct because all expected top-level keys exist.

So `schema_keys_rate = 1.0` for `mse` should not be read as full schema
fidelity.

### `key_overlap_mean` is a narrow proxy for MSE

The current primary-key metric for MSE compares only `appearance.text`.

That can understate partial success if another MSE domain is predicted correctly
while `appearance` remains null.

In this run the qualitative examples still show a broader collapse, so this
metric caveat does not change the main conclusion. It just means the MSE
evaluation could be improved.

## Conclusion

`v2-multitemplate-sft` demonstrated that schema injection plus SFT can work for:

- seen template families
- evidence-grounded extraction
- transcript-style transfer within a known schema

It did not demonstrate robust zero-shot transfer to a new clinical output
ontology.

The key lesson from v2 is:

> schema text alone was sufficient for format control, but not sufficient for
> semantic transfer into unseen clinical field vocabularies under a conservative
> null policy.

## Recommended Next Step

The cleanest next experiment is `v2.1`:

- add `mse` into the training mix
- hold out a genuinely new template such as discharge summary or progress note
- rerun the same Unsloth setup
- compare v2 and v2.1 side by side

That would separate two questions more honestly:

- can the model learn a new ontology once supervised examples exist
- how much zero-shot transfer remains to a truly unseen template family

## Artifact References

- prompt assembly: `src/prompts.py`
- template registry: `src/data_generation/templates/__init__.py`
- MSE spec: `src/data_generation/templates/mse.py`
- referral specs: `src/data_generation/templates/referral.py`
- canonical notebook: `notebooks/v2_multitemplate_sft_unsloth.ipynb`
- saved run output: `notebooks/v2-multitemplate-sft-unsloth-200run-output.ipynb`
