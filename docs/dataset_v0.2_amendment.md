# Dataset v0.2 Amendment

## Status

Pretraining design amendment.

This amendment was defined after dataset v0.1 failed the token-level
training-support audit and before any model training or held-out
behavioral-result inspection.

## 1. Reason for amendment

Dataset v0.1 used one opaque composite request register token per
complete register configuration.

The two request configurations withheld for compositional-OOD
evaluation therefore also withheld their corresponding model-facing
tokens:

```text
<C03>
<C04>
