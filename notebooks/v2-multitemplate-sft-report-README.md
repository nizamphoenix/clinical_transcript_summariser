# v2-multitemplate-sft Report

## Definitions
- schema_valid (0–1): did the model's output parse as JSON AND contain all required keys with the right nesting? 1 = yes, 0 = no. Averaged over the eval set. So schema 0.0 on v2 MSE means every single MSE output was structurally broken (e.g. missing the appearance.text nesting).
- key_overlap / overlap (0–1): of the gold keys, how much of the gold content appears in the model's output, measured by generalized token overlap on the values. overlap 0.01 means the populated fields share almost no tokens with gold — i.e. the model emitted near-empty values.
- grounding (0–1): of the spans the model emitted, what fraction are exact substrings of the transcript? 1 = nothing hallucinated, lower = some emitted text isn't in the transcript.
Plus two diagnostics:
- all_null_rate: fraction of outputs where every field is null/empty. v2 MSE was 0.6 — 60% of outputs were fully collapsed.
- ungrounded_span_rate (proposed, not yet in harness): fraction of emitted spans not found in transcript. Needed to detect hallucination honestly.
"The verifier" is just these metrics packaged as one function: verifier(transcript, schema, output) → {schema_valid, grounding, key_overlap, ungrounded_span_rate}. Same code, two uses:
1. As an evaluator in the eval notebook (what we do now).
2. As a reward signal for DPO (rank candidates) and GRPO (per-rollout reward).
That dual use is the contribution. The verifier is the bridge between eval and post-training. It's also the "domain-tailored novel technique" line from the JD — generic LM eval doesn't know what a clinical transcript span is; ours does.

#### SFT (Supervised Fine-Tuning) — teaches the shape of the answer.
- What it solves: getting the model to produce outputs that look like your gold examples. Format, vocabulary, schema structure, domain phrasing.
- How: show input → gold output pairs. Loss = "how different was the model's next-token prediction from the gold token?" Updates weights to close that gap.
- Strength: cheap, stable, the only thing that works when the model has no idea what the task even is. You need SFT to get the model into the right ballpark.
- Weakness: it can only imitate. It can't tell the model "this output was better than that one" — only "this is the one right answer." So it can't fix subtle behavioural problems (collapse, hallucination) and can't optimise a quality signal beyond token match.

#### DPO (Direct Preference Optimisation) — teaches which output is better when two are plausible.
- What it solves: behavioural preferences between similar-looking outputs. "Populate when grounded, abstain when not." "Don't hallucinate when uncertain." "Prefer concise over verbose."
- How: show input → (chosen, rejected) pairs. Loss pushes chosen up, rejected down, relative to a frozen reference copy of the model (the reference keeps it from drifting too far).
- Strength: directly optimises the failure mode you care about, without needing a reward model or rollouts. Cheap, close to SFT in tooling.
- Weakness: needs the model to already roughly do the task — it's polishing, not bootstrapping. And it inherits whatever bias is in your preference pairs (this is your 2b point).

#### GRPO (Group Relative Policy Optimisation) — teaches the model to maximise a numerical reward online.
- What it solves: same behavioural problems as DPO, but with a decomposable reward you can tune. e.g. trade off schema validity vs grounding vs coverage explicitly.
- How: at each step, sample N outputs for the same prompt, score each with a reward function, update weights to favour the higher-scoring ones within that group. No separate reward model needed if your reward is programmatic (ours is — it's the verifier).
- Strength: most flexible. You can add/remove reward terms without regenerating a dataset. Current SOTA recipe (DeepSeek-R1-Zero used this).
- Weakness: expensive (N rollouts per prompt), unstable, prone to reward hacking. Needs careful reward design.

#### Do we still need SFT first?
Yes, and here's the friendly intuition. DPO and GRPO both rely on the model already producing reasonable candidates to choose between or to score. If you start from a base model that has never seen clinical transcripts in JSON form, every candidate it samples will be garbage — DPO has nothing to prefer between, GRPO has nothing to reward. SFT is what makes "reasonable candidate generation" possible. Then DPO/GRPO push those reasonable candidates toward great ones.
The standard pipeline: SFT (learn the shape) → DPO or GRPO (learn the preference / reward) → optionally iterate.

#### One-sentence summaries for the interview:
- SFT teaches the model what an answer looks like.
- DPO teaches the model which of two answers is better.
- GRPO teaches the model how to maximise a quality score it can compute.
- **None of them** teach the model a new ontology it has never seen — that's still a data-coverage problem (Problem 1).

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


## Conclusion. 
- v1 only saw SOAP, failed on referral / MSE.
- v2 saw SOAP + referral_a, failed on MSE (unseen ontology).
- v2.1-lite added MSE training data → MSE improved. But this solved by training on it, not by generalising to it. The true zero-shot-to-unseen-ontology ceiling is still there — referral_b is style transfer, not unseen ontology.

"SFT got me to schema-valid output on trained templates. It produced two failure modes: conservative collapse on near-OOD prompts, and a latent hallucination risk if I'd pushed coverage harder. SFT can't fix either because token-level loss doesn't see schema validity or evidence grounding. I built a verifier (schema + grounding + overlap) and used it as a preference signal for DPO, then as a reward for GRPO. DPO addressed collapse; GRPO let me decompose the reward and explicitly trade collapse against hallucination. Neither addresses unseen-ontology zero-shot — that's a data-coverage and constrained-decoding problem, which I'd attack next."

"On in-distribution templates v2 behaved fine. On the unseen MSE template it didn't hallucinate — it over-abstained. Some of those nulls were correct (no evidence), some were wrong (evidence present, model still abstained). SFT can't separate those two cases because token loss against gold targets doesn't model evidence presence."

### Problem 2: SFT failure modes in schema injection
It is behavioural failure modes that SFT produces or can't fix, even when the model has seen the schema. Three distinct sub-failures, all observed or latent in your runs:
2a. Conservative collapse (observed in v2 on MSE)
- v2 saw SOAP + referral_a, then was shown MSE at inference.
- Output: schema-shaped but fields mostly null or empty strings. Schema validity 0.0 because it sometimes drops required nesting; key overlap 0.0104; all_null_rate 0.6.
- Why SFT caused this: trained on two templates that share some fields, the model learned "when uncertain about a field, abstain" as a safe policy. SFT reinforces the average of its training targets; when the test prompt is off-distribution, the safest average is "say nothing."
- SFT can't directly fix this without more data, because there's no gradient signal that says "abstention here was wrong."
2b. Over-population / hallucination (latent risk, not yet observed)
- A model trained to "always fill the schema" will invent plausible content when the transcript lacks evidence. e.g. transcript says nothing about affect, model writes "affect": "patient appeared euthymic" because that's what training examples looked like.
- SFT can't distinguish "populate because grounded" from "populate because it looks like training." The loss only sees the gold target.
2c. Format-vs-content tradeoff
- SFT optimises token-level cross-entropy. A model can score well on token loss while being subtly ungrounded, or score badly on token loss while being semantically correct (paraphrased evidence). SFT has no notion of "this span exists in the transcript" as a first-class objective.
In your numbers: v2 → v2.1-lite on MSE went from schema 0.0 / overlap 0.01 (collapse) to schema 1.0 / overlap 0.14 (usable). But the fix was more SFT data, not a behavioural fix. The collapse failure mode still lives in any future unseen template until you train on it.

#### What DPO and GRPO actually fix?
**DPO** fixes 2a and 2c directly. It mitigates 2b only if rejected examples include hallucinations.
- 2a: rejected = collapsed outputs, chosen = populated grounded outputs → model learns "abstaining when evidence exists is wrong."
- 2c: chosen/rejected ranked by verifier (schema + grounding + overlap), not token loss → optimises the actual quality signal, not a proxy.
- 2b: only fixed if you include hallucinated samples as rejected. Otherwise DPO can make 2b worse.



**GRPO** fixes the same set, but online and with a multi-term reward.
- Generates multiple samples per prompt, ranks them by verifier reward, updates policy to prefer higher-reward samples within the group.
- Advantage over DPO: the reward is decomposable (schema_valid + grounding − hallucination_penalty + coverage), so you can directly trade off failure modes by tuning weights.
- Advantage in the room: GRPO is the DeepSeek-R1-Zero recipe. Name-droppable, current, and the role explicitly says "RLFT."

Neither fixes **Problem 1 (unseen-ontology zero-shot)**.


