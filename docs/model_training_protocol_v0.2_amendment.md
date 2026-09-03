# Model Training Protocol v0.2 Amendment

## Status

This amendment was defined before model training.

It updates the original model-training protocol only where dataset v0.2
changed the model-facing representation.

All other provisions of `model_training_protocol.md` remain in force.

## Primary dataset

Primary experiments use dataset v0.2.

Dataset fingerprint:

ff1315604a48d991a97af8a1b8ba8749f5f9bff94614263a155d216c7770e8c7

Dataset v0.1 is rejected and must not be used for primary training.

## Request register controls

Request contexts no longer expose opaque composite tokens
`<C00>` through `<C07>`.

Instead, each request contains three entangled control tokens drawn
from:

<RC1_0> <RC1_1>
<RC2_0> <RC2_1>
<RC3_0> <RC3_1>

Assertion contexts continue to use:

<C08> <C09> <C10> <C11>

The linguistic composite codes remain available as researcher-facing
metadata.

## Vocabulary

The deterministic closed model vocabulary contains 154 tokens.

The tokenizer fingerprint must be recorded with model checkpoints.

## Sequence length

The longest v0.2 request sequence contains 14 tokens.

The model maximum sequence length remains 16.

No architecture change is required.

## Model architecture

The primary starting architecture remains:

layers: 2
attention heads: 4
model dimension: 128
head dimension: 32
MLP hidden dimension: 512
maximum sequence length: 16
activation: GELU
normalization: pre-LayerNorm
position representation: learned absolute positional embeddings

With the v0.2 vocabulary, the implementation contains 438,272 trainable parameters.

## Development separation

Training and validation data may be inspected during model development.

The following remain unavailable for model selection or hyperparameter
development:

iid_test
compositional_ood_test
lexical_ood_test

The final training configuration must be frozen before those splits are
evaluated.
