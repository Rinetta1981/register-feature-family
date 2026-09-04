# Behavioral Gate v0.2: Confirmatory Result

## Status

Confirmatory behavioral evaluation completed.

Primary result: **behavioral gate failed**.

Passing primary seeds: **0 of 3**

Required passing seeds: **2 of 3**

Under the frozen research protocol, the planned primary mechanistic
analysis is therefore not treated as supported by behavioral validity.

This result is retained as a negative result.

## Research question

The broader project asks whether a transformer represents linguistic
register as one general formal–informal direction or as a family of
causally separable features.

Before interpreting internal representations, the study imposed a
behavioral validity gate. The model first had to demonstrate that it had
learned the task and could generalize the relevant register distinctions
beyond the training distribution.

## Confirmatory safeguards

The following were frozen before held-out behavioral evaluation:

- dataset version and fingerprint;
- model architecture;
- tokenizer;
- primary model seeds 0, 1, and 2;
- training hyperparameters;
- validation-only checkpoint selection;
- behavioral metrics;
- behavioral thresholds;
- 2-of-3 seed decision rule.

The IID, compositional-OOD, and lexical-transfer-OOD results were not
used for hyperparameter selection.

A scoring implementation issue involving multi-token semantic
realizations was discovered by CI tests before held-out exposure. It was
corrected and documented as a pre-exposure amendment without changing
the dataset, thresholds, model seeds, training configuration, or
held-out splits.

## Frozen dataset

Dataset version: 0.2

Dataset fingerprint:

ff1315604a48d991a97af8a1b8ba8749f5f9bff94614263a155d216c7770e8c7

Held-out split sizes:

- IID test: 120 examples
- compositional OOD test: 216 examples
- lexical-transfer OOD test: 360 examples

Tokenizer fingerprint:

f1cf46bd23a6b4870bfdfc4192cff98b7d7fc20706052c2a1aee9c278b793eef

## Development behavior

Before confirmatory exposure, all three model seeds achieved validation
exact-match accuracy of 1.0.

The frozen training configuration was therefore retained without further
hyperparameter tuning.

## Confirmatory results

| Seed | IID exact match | IID content | IID register | Compositional-OOD content | Compositional-OOD register | Lexical-OOD content | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 100% | 100% | 100% | 99.1% | 0% | 7.5% | Fail |
| 1 | 100% | 100% | 100% | 100% | 0% | 12.5% | Fail |
| 2 | 100% | 100% | 100% | 100% | 0% | 2.2% | Fail |

No primary seed passed the frozen behavioral gate.

## Result 1: IID performance was not the problem

All three independently initialized models achieved perfect performance
on the IID test set:

- content accuracy: 100%;
- register accuracy given correct content: 100%;
- full exact-match accuracy: 100%.

The failed gate therefore cannot be explained as a general failure to
fit the task.

This creates a useful contrast between successful in-distribution
learning and failed out-of-distribution generalization.

## Result 2: semantic content generalized compositionally, register did not

On the compositional-OOD split, content accuracy was:

- seed 0: 214 / 216 = 99.1%;
- seed 1: 216 / 216 = 100%;
- seed 2: 216 / 216 = 100%.

Despite this near-perfect preservation of semantic content, register
accuracy given correct content was 0% for every seed.

Exact-match accuracy was also 0% for every seed.

This is a structured failure rather than undifferentiated model error:
the models generally preserved what was being expressed while failing
to produce the unseen combination of register features.

Diagnostic component metrics further show that different register
components behaved differently across seeds.

For seed 0, lexical-formality realization was 100% among
content-correct compositional examples, while directness was 0% and
mitigation was approximately 30.4%.

For seed 1, lexical formality was 100%, while directness and mitigation
were both 0%.

For seed 2, lexical formality was approximately 36.6%, directness
approximately 45.8%, and mitigation 50%.

These component results are diagnostic rather than confirmatory
mechanistic evidence. They suggest that perfect IID behavior did not
correspond to a reliably recombinable register representation.

## Result 3: lexical transfer failed primarily at semantic transfer

Lexical-transfer-OOD content accuracy was low for every seed:

- seed 0: 27 / 360 = 7.5%;
- seed 1: 45 / 360 = 12.5%;
- seed 2: 8 / 360 = 2.2%.

Among the small subset of lexical-transfer examples whose semantic
content was correct, register accuracy was 100%.

Those conditional register figures have very small denominators and
should not be interpreted as strong evidence of lexical-transfer
register generalization.

The dominant lexical-transfer failure was instead the model's inability
to use lexical equivalences learned during familiarization when
generating register-conditioned targets for held-out semantic content.

## Behavioral-gate decision

The frozen seed-level gate required all of the following:

- IID content accuracy >= 0.95;
- compositional-OOD content accuracy >= 0.90;
- lexical-transfer-OOD content accuracy >= 0.90;
- IID register accuracy given correct content >= 0.90;
- compositional-OOD register accuracy given correct content >= 0.75.

The overall gate required at least two of three seeds to pass.

Observed passing seeds:

0 / 3

Therefore:

**The behavioral gate failed.**

The primary mechanistic analysis is not used to support the claim that
the trained models learned the intended causally separable register
feature family.

## Scientific interpretation

The result rules out a simple success narrative.

The models learned the IID mapping extremely well but did not exhibit
the required forms of generalization.

The strongest descriptive finding is a dissociation:

1. compositional semantic content generalization was nearly perfect
   while compositional register realization failed completely; and

2. lexical-transfer failures were dominated by semantic-transfer
   failure rather than register errors among successful semantic cases.

These results are compatible with several possible internal
representations, including memorized or partially factorized mappings.
Behavioral results alone do not distinguish among those mechanisms.

Because the preregistered validity gate failed, this study does not make
a confirmatory mechanistic claim about a one-dimensional register axis
versus a causally separable feature family.

## Why the negative result matters

Perfect IID accuracy can create the appearance that a small transformer
has learned a structured latent system.

Here, held-out interventions on the behavioral distribution show that
this conclusion would have been premature.

The experiment therefore demonstrates the value of behavioral
generalization gates before mechanistic interpretation.

It also identifies two concrete failure modes that can guide a future
study:

- failure to recombine register dimensions despite preserved semantic
  content;
- failure to transfer familiarized lexical equivalences into
  register-conditioned generation.

## Future work

A future v0.3 experiment should be treated as a new study rather than as
a continuation of the v0.2 confirmatory test.

Because the v0.2 held-out results have now been observed, they must not
be reused as unseen confirmatory evidence after model or dataset
redesign.

A clean follow-up would require new held-out assignments or a newly
generated frozen dataset.

Candidate questions for v0.3 include:

- what training signal is sufficient for lexical equivalence to transfer
  into register-conditioned generation;
- whether alternative distributed control encodings improve
  compositional recombination;
- whether training regimes produce partially factorized internal
  representations before full behavioral compositionality emerges;
- whether causal feature interventions are meaningful only after a
  stronger behavioral generalization criterion is satisfied.

These are future hypotheses, not conclusions from v0.2.

## Reproducibility

The repository preserves:

- deterministic synthetic-data generation;
- dataset fingerprints and manifests;
- closed-vocabulary tokenization;
- explicit transformer implementation;
- typed training and evaluation code;
- automated Ruff, mypy, and pytest checks;
- validation-only development;
- frozen training configuration;
- pre-exposure behavioral metric definitions;
- documented pre-exposure amendments;
- prediction-level confirmatory outputs;
- three independently initialized primary seeds.

This separation between development, confirmatory evaluation, and
post-result interpretation is part of the experimental result.
