# Register Is a Feature Family, Not a Dial?

![CI](https://github.com/Rinetta1981/register-feature-family/actions/workflows/ci.yml/badge.svg)

A controlled transformer experiment asking whether linguistic register is
represented as one general formal–informal direction or as a family of
separable features.

**Study status: v0.2 complete. Confirmatory behavioral gate failed.**

The project therefore stops short of making the planned primary
mechanistic claim.

That stopping decision is part of the result.

---

## Executive summary

A small decoder-only transformer was trained on a synthetic language in
which register is generated from independently specified dimensions such
as lexical formality, directness, mitigation, and epistemic stance.

Before analyzing internal representations, the study imposed a frozen
behavioral validity gate.

The model had to demonstrate that it had learned more than an
in-distribution lookup table.

Three independently initialized models achieved:

- 100% IID exact-match accuracy;
- 99.1–100% semantic-content accuracy on compositional OOD;
- 0% register accuracy on compositional OOD;
- only 2.2–12.5% semantic-content accuracy on lexical-transfer OOD.

No seed passed the preregistered behavioral gate.

Because the validity gate failed, the planned primary mechanistic
analysis is not treated as confirmatory evidence about whether register
is represented as a single direction or a causally separable feature
family.

The main empirical result is instead a structured generalization
failure:

**semantic content recombined almost perfectly under compositional OOD,
while register did not recombine at all.**

A second failure appeared under lexical transfer:

**lexical equivalences learned during familiarization largely failed to
transfer into register-conditioned generation.**

---

## Research question

Does a transformer represent linguistic register as:

1. one approximately one-dimensional formal–informal direction; or
2. a family of partially independent features that can be recombined and
   causally manipulated separately?

The intended feature family includes:

- lexical formality;
- syntactic directness;
- politeness mitigation;
- epistemic stance;
- speaker–addressee power relation as a contextual variable.

The central methodological choice was to require behavioral evidence of
generalization before interpreting internal representations.

---

## Why require a behavioral gate?

Mechanistic interpretability can produce compelling-looking internal
structure even when a model has learned the wrong behavioral rule.

A probe, direction, sparse feature, attention pattern, or intervention is
not automatically evidence that the model learned the intended latent
factorization.

This project therefore separates two questions:

**Behavioral validity**

Did the model learn the intended distinctions in a way that generalizes
outside the training distribution?

**Mechanistic explanation**

If so, how are those distinctions represented and causally implemented?

The second question was conditional on passing the first.

That condition was not met in v0.2.

---

## Experimental design

### Synthetic language

The study uses a fully controlled synthetic language rather than natural
English for the primary experiment.

This makes the ground-truth semantic and register variables known by
construction.

The frozen inventory contains:

- 16 request meanings;
- 16 assertion meanings;
- 32 semantic-content families;
- 3 lexical realizations per content family;
- 96 learned synthetic lexical forms.

Synthetic lexical items are generated deterministically.

Natural-language replication was planned only as a later stage, after
the controlled experiment established behavioral validity.

### Request register dimensions

Requests vary along three binary register dimensions:

- lexical formality;
- directness;
- mitigation.

This yields 8 possible register combinations.

Two combinations are held out globally from register-conditioned
training for the compositional-OOD test.

### Assertion register dimensions

Assertions vary along:

- lexical formality;
- epistemic stance.

This yields 4 register combinations.

### Entangled request controls

Dataset v0.2 does not directly expose the intended feature
factorization to the model.

For request features L, D, and M, the model receives three distributed
control bits:

- RC1 = L XOR D
- RC2 = D XOR M
- RC3 = L XOR D XOR M

All individual control-token values occur during training.

The complete control combinations corresponding to the two
compositional-OOD register configurations do not.

This makes successful OOD behavior require recombination rather than
learning an unseen token embedding.

---

## Frozen dataset

Primary dataset version:

`0.2`

Dataset fingerprint:

`ff1315604a48d991a97af8a1b8ba8749f5f9bff94614263a155d216c7770e8c7`

Confirmatory records:

`1656`

Split sizes:

| Split | Examples |
| --- | ---: |
| Train | 840 |
| Validation | 120 |
| IID test | 120 |
| Compositional OOD test | 216 |
| Lexical-transfer OOD test | 360 |

Lexical familiarization adds 96 training records.

The dataset generator, split assignments, manifests, fingerprints, and
audits are deterministic and stored in the repository.

---

## Model

The primary model is a small explicit decoder-only transformer.

Architecture:

- 2 transformer layers;
- 4 attention heads;
- model width 128;
- head dimension 32;
- MLP width 512;
- GELU activation;
- pre-LayerNorm;
- learned absolute positional embeddings;
- separate embedding and unembedding matrices;
- maximum sequence length 16;
- 438,272 trainable parameters.

The implementation exposes attention internals directly so that later
causal and representation analyses would be possible if the behavioral
gate were passed.

---

## Training protocol

Primary model seeds:

- 0
- 1
- 2

Training occurs in two stages.

### Stage A: lexical familiarization

The model first learns the synthetic lexical inventory independently of
the register-conditioned task.

### Stage B: register-conditioned training

The model then trains on register-conditioned examples, with lexical
familiarization examples rehearsed during training.

Development uses only:

- training data;
- familiarization data;
- validation data.

IID, compositional-OOD, and lexical-transfer-OOD test results are not
used for hyperparameter selection or checkpoint selection.

Frozen hyperparameters:

| Hyperparameter | Value |
| --- | ---: |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Gradient clip norm | 1.0 |
| Familiarization epochs | 150 |
| Register epochs | 80 |
| Minimum register epochs | 20 |
| Early-stopping patience | 12 |
| Early-stopping minimum delta | 0.00001 |

All three development seeds achieved validation exact-match accuracy of
1.0 before the training configuration was frozen.

---

## Frozen behavioral gate

The confirmatory gate was specified before held-out behavioral results
were inspected.

A seed had to satisfy all of the following:

- IID content accuracy >= 95%;
- compositional-OOD content accuracy >= 90%;
- lexical-transfer-OOD content accuracy >= 90%;
- IID register accuracy given correct content >= 90%;
- compositional-OOD register accuracy given correct content >= 75%.

The experiment-level gate required at least 2 of 3 primary seeds to
pass.

Observed result:

**0 of 3 seeds passed.**

---

## Confirmatory results

| Seed | IID exact match | Comp-OOD content | Comp-OOD register | Lexical-OOD content | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 100% | 99.1% | 0% | 7.5% | Fail |
| 1 | 100% | 100% | 0% | 12.5% | Fail |
| 2 | 100% | 100% | 0% | 2.2% | Fail |

### IID performance

All three seeds achieved:

- 100% content accuracy;
- 100% register accuracy given correct content;
- 100% exact-match accuracy.

The failed gate is therefore not explained by ordinary failure to fit the
task.

### Compositional OOD

Semantic-content accuracy remained almost perfect:

- seed 0: 214 / 216;
- seed 1: 216 / 216;
- seed 2: 216 / 216.

Register accuracy given correct content was:

- seed 0: 0%;
- seed 1: 0%;
- seed 2: 0%.

Full exact-match accuracy was also 0% for every seed.

The models therefore generally preserved what was being expressed while
failing to produce the held-out combination of register features.

### Register-component diagnostics

The failure was not identical across all register dimensions.

| Seed | Lexical formality | Directness | Mitigation |
| --- | ---: | ---: | ---: |
| 0 | 100% | 0% | 30.4% |
| 1 | 100% | 0% | 0% |
| 2 | 36.6% | 45.8% | 50.0% |

These are descriptive behavioral diagnostics.

They are not evidence that the corresponding internal features are
mechanistically independent.

### Lexical-transfer OOD

Semantic-content accuracy collapsed:

- seed 0: 27 / 360 = 7.5%;
- seed 1: 45 / 360 = 12.5%;
- seed 2: 8 / 360 = 2.2%.

Among the small subset of examples where semantic content was correct,
register realization was also correct.

Those conditional results have very small denominators and should not be
treated as strong evidence of successful lexical-transfer
generalization.

The dominant failure is instead that lexical equivalences learned during
familiarization did not reliably transfer into register-conditioned
generation.

---

## Scientific decision

The preregistered behavioral gate failed.

Therefore the project does **not** use the planned primary mechanistic
analysis to claim that the trained transformer represents register as a
single axis or as a causally separable feature family.

This is deliberate.

The internal organization of a model that does not exhibit the required
behavioral generalization may still be interesting, but interpreting it
as the intended latent system would risk answering a different question
from the one the experiment was designed to test.

The negative result is retained rather than tuned away.

---

## What the result does support

The v0.2 study supports a narrower descriptive conclusion.

Perfect IID performance did not imply the intended structured
generalization.

Two qualitatively different failures appeared:

1. **Compositional failure:** semantic content generalized while register
   recombination failed.
2. **Lexical-transfer failure:** familiarized semantic equivalences did
   not reliably transfer into register-conditioned generation.

These failures constrain the design of future mechanistic experiments.

They do not by themselves identify the underlying internal mechanism.

---

## What this project does not claim

This repository does not claim that:

- register is definitively represented as a single direction;
- register is definitively represented as a separable feature family;
- the diagnostic component accuracies reveal causal internal features;
- perfect IID performance demonstrates latent compositionality;
- post-hoc tuning on the exposed v0.2 test sets would constitute a new
  confirmatory result.

The behavioral evidence does not justify those conclusions.

---

## Research safeguards and amendments

### Dataset v0.1 rejection

An earlier frozen dataset version was rejected before model training.

The original request design represented each full register configuration
with an opaque composite token.

The two compositional-OOD request codes never appeared during training,
which meant their embeddings would have been unsupported.

That would have confounded compositional generalization with an untrained
input-representation problem.

The defect was documented, v0.1 was retained historically, and v0.2 was
created before training.

### Pre-exposure metric amendment

During CI testing of the v0.2 behavioral evaluator, a validation example
revealed that a target can contain multiple familiarized lexical forms
from the same semantic-content family.

The initial scoring implementation incorrectly assumed exactly one such
form.

The implementation and protocol were corrected before any held-out
behavioral evaluation was launched.

No test result had been inspected, and no dataset, threshold, seed,
training hyperparameter, or held-out split was changed.

---

## Reproducibility

The repository includes:

- deterministic dataset generation;
- frozen dataset fingerprints;
- explicit split audits;
- a closed-vocabulary tokenizer;
- explicit sequence encoding;
- a custom decoder-only transformer;
- deterministic training batches;
- validation-only development;
- three frozen primary model seeds;
- model and tokenizer fingerprints;
- checkpoint metadata;
- prediction-level confirmatory outputs;
- automated Ruff checks;
- strict mypy checks;
- pytest test coverage;
- manually triggered GitHub Actions training and evaluation workflows.

To install the project locally:

    python -m pip install -e ".[dev]"

Run the quality checks:

    ruff check .
    mypy src
    pytest -q

Train one development seed:

    python scripts/train_development_model.py 0 artifacts/development-seed-0

Reproduce the now-exposed v0.2 behavioral evaluation:

    python scripts/evaluate_behavioral_gate.py artifacts/behavioral-evaluation

The latter command reproduces an already exposed confirmatory result. It
must not be treated as a new blinded evaluation.

---

## Repository guide

Key research documents:

- `docs/research_protocol.md` — overall experimental protocol;
- `docs/data_specification.md` — dataset schema and construction;
- `docs/dataset_splits.md` — split design;
- `docs/lexical_transfer_familiarization.md` — lexical-transfer design;
- `docs/dataset_v0.1_rejection.md` — rejected first dataset design;
- `docs/dataset_v0.2_amendment.md` — corrected dataset design;
- `docs/frozen_dataset_v0.2.md` — primary frozen dataset;
- `docs/model_training_protocol.md` — training design;
- `docs/model_training_protocol_v0.2_amendment.md` — v0.2 training amendment;
- `docs/frozen_behavioral_evaluation_protocol.md` — frozen behavioral gate;
- `docs/behavioral_gate_v0.2_result.md` — confirmatory result and interpretation.

Key implementation modules:

- `src/register_feature_family/experimental_generator.py`
- `src/register_feature_family/request_controls.py`
- `src/register_feature_family/dataset_v02.py`
- `src/register_feature_family/tokenizer.py`
- `src/register_feature_family/sequence_encoding.py`
- `src/register_feature_family/model.py`
- `src/register_feature_family/training.py`
- `src/register_feature_family/behavioral_evaluation.py`

---

## Study history

| Version | Status | Reason |
| --- | --- | --- |
| v0.1 | Rejected before training | Compositional-OOD request codes had no training support |
| v0.2 | Completed | Behavioral gate failed 0/3 seeds |

Both outcomes are retained.

The failed v0.1 design is part of the methodological record rather than
being silently discarded.

---

## Future work

A future v0.3 should be treated as a new experiment.

Because the v0.2 test results have been exposed, a redesigned model or
dataset should use fresh held-out assignments before making new
confirmatory claims.

High-value follow-up questions include:

- What training signal is sufficient for lexical equivalence to transfer
  into register-conditioned generation?
- Why can semantic content recombine while register features fail to
  recombine?
- Do partially factorized internal representations emerge before
  behavioral compositionality?
- Which training distributions produce genuinely recombinable register
  variables?
- Once behavioral validity is established, do causal interventions reveal
  one dominant register direction or multiple separable mechanisms?

Those questions are motivated by v0.2.

They are not conclusions from it.

---

## Bottom line

The transformer learned the IID task perfectly.

It did not learn the form of structured generalization required by the
experiment.

Rather than interpreting internal features anyway, the study stopped at
its preregistered behavioral gate.

For mechanistic interpretability, that distinction matters.
