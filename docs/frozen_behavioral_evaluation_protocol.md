# Frozen Behavioral Evaluation Protocol

## Status

Frozen before confirmatory held-out behavioral evaluation.

Freeze date: 2026-09-04

The IID test, compositional-OOD test, and lexical-transfer-OOD test
results had not been inspected when this protocol was frozen.

## Pre-exposure implementation amendment

During CI testing of the behavioral evaluator, a validation example
demonstrated that a generated target can contain more than one
familiarized lexical surface form belonging to the same semantic content
family.

The initial evaluator implementation incorrectly assumed that a target
contained exactly one familiarized surface form.

No held-out behavioral evaluation had been launched and no IID,
compositional-OOD, or lexical-transfer-OOD model result had been
inspected when this issue was detected.

The content metric was therefore corrected before confirmatory exposure.

The correction does not alter:

- the frozen dataset;
- the model seeds;
- the training configuration;
- the behavioral thresholds;
- the 2-of-3 seed rule;
- the held-out splits.

It only makes the scoring implementation agree with the synthetic
grammar already present in the frozen dataset.

## Primary dataset

Dataset version: 0.2

Dataset fingerprint:

ff1315604a48d991a97af8a1b8ba8749f5f9bff94614263a155d216c7770e8c7

Confirmatory records: 1656

Held-out confirmatory splits:

- IID test: 120 records
- compositional OOD test: 216 records
- lexical-transfer OOD test: 360 records

## Primary model seeds

The primary model initialization seeds are:

- 0
- 1
- 2

No additional primary seeds will be selected after held-out results are
observed.

## Development results used to freeze training

Seed 0:

- validation exact match: 1.0
- best validation loss: 0.0002699325216924894
- best validation epoch: 79
- familiarization final loss: 0.004125788186987241

Seed 1:

- validation exact match: 1.0
- best validation loss: 0.00026465166803590315
- best validation epoch: 79
- familiarization final loss: 0.003787167059878508

Seed 2:

- validation exact match: 1.0
- best validation loss: 0.00026001566364261395
- best validation epoch: 78
- familiarization final loss: 0.00365504032621781

All three development models contained 438272 trainable parameters.

All three development runs used source revision:

bb2aa4048a698dab3521976ced42c8929d3b6659

Tokenizer fingerprint:

f1cf46bd23a6b4870bfdfc4192cff98b7d7fc20706052c2a1aee9c278b793eef

## Frozen training configuration

The following configuration is frozen for confirmatory behavioral
evaluation:

- batch size: 64
- learning rate: 0.001
- weight decay: 0.01
- gradient clipping norm: 1.0
- lexical familiarization epochs: 150
- register training epochs: 80
- minimum register epochs before early stopping: 20
- early-stopping patience: 12
- early-stopping minimum delta: 0.00001

Training uses target-only causal language-model loss.

Stage A trains on lexical familiarization records.

Stage B trains on register-conditioned training records together with
one copy of the lexical familiarization set per epoch as rehearsal.

Checkpoint selection and early stopping use validation loss only.

No held-out test split is used for training, checkpoint selection,
early stopping, or hyperparameter selection.

## Behavioral metric definitions

### Content accuracy

Each synthetic semantic content has three lexical surface forms learned
during lexical familiarization.

A generated target may contain more than one familiarized surface form
from its semantic content family.

For each evaluation example, the expected number of familiarized lexical
tokens is determined from the frozen expected target.

A prediction is content-correct when:

- it contains the same number of familiarized lexical tokens as the
  expected target; and
- every generated familiarized lexical token belongs to the semantic
  content family identified by the example context.

This definition treats substitutions among lexical forms belonging to
the correct semantic family as content-correct while allowing register
metrics to distinguish whether the appropriate lexical realization was
used.

Missing lexical tokens, extra lexical tokens, or lexical tokens from a
different semantic family are content errors.

Content accuracy is the proportion of all examples that are
content-correct.

### Register accuracy given correct content

Register accuracy is evaluated only among examples whose semantic
content is correct.

For a content-correct example, the register realization is correct when
the complete generated target exactly matches the expected target.

This captures all register-bearing choices, including lexical formality
and the applicable grammatical register markers.

This conditional definition separates semantic-content failure from
register-realization failure.

### Joint exact-match accuracy

Joint exact-match accuracy is the proportion of all examples for which
the complete generated target exactly matches the expected target.

This is reported in addition to the separated content and register
metrics.

### Diagnostic component metrics

For content-correct examples, the evaluator also reports:

- lexical-formality realization accuracy;
- request directness-marker accuracy;
- request mitigation-marker accuracy;
- assertion epistemic-stance-marker accuracy.

Lexical-formality realization is correct when the ordered sequence of
familiarized lexical forms in the generated target matches the ordered
sequence in the expected target.

These are diagnostic metrics and do not replace the preregistered
behavioral gate.

## Frozen behavioral gate

A seed passes the behavioral gate only if all of the following are
true:

- IID content accuracy >= 0.95
- compositional-OOD content accuracy >= 0.90
- lexical-transfer-OOD content accuracy >= 0.90
- IID register accuracy given correct content >= 0.90
- compositional-OOD register accuracy given correct content >= 0.75

Lexical-transfer register accuracy is reported but does not have an
additional confirmatory threshold.

The overall behavioral gate passes if at least 2 of the 3 primary seeds
pass the seed-level gate.

If fewer than 2 of the 3 primary seeds pass, the primary mechanistic
analysis is not treated as supported by the behavioral gate.

Null and failed behavioral results are retained and reported.

## Confirmatory exposure rule

The evaluator, its metric definitions, thresholds, frozen training
configuration, and tests must be committed to the repository and pass
CI before the first held-out behavioral evaluation is launched.

After held-out results are exposed, the primary behavioral metrics,
thresholds, dataset, model seeds, and frozen training configuration will
not be changed in response to those results.
