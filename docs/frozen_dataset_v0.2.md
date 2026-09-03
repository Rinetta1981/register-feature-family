# Frozen Dataset v0.2

## Status

Dataset v0.2 is frozen and approved for model training.

The dataset was frozen after dataset v0.1 was rejected by a
pretraining token-support audit and before any model training or
held-out behavioral-result inspection.

## Dataset identity

Dataset version: v0.2

Dataset seed: 0

Neutral familiarization VAR0/VAR1 swap: false

## Record counts

Register-conditioned confirmatory records: 1,656

Neutral lexical-familiarization records: 96

Confirmatory split counts:

| Split | Records |
|---|---:|
| train | 840 |
| validation | 120 |
| iid_test | 120 |
| compositional_ood_test | 216 |
| lexical_ood_test | 360 |

## File fingerprints

### confirmatory.jsonl

Records: 1,656

SHA-256:

fea78eb27afb4a9feec162ee3f1a98d61a75dd95a66b92da19c5593deb6a61ef

### lexical_familiarization.jsonl

Records: 96

SHA-256:

4f8a212f3585c8af88154952d7f112e33bd6de1de674bd851ce8088f6d2f8f0a

The lexical-familiarization file is byte-identical to dataset v0.1.

## Complete dataset fingerprint

ff1315604a48d991a97af8a1b8ba8749f5f9bff94614263a155d216c7770e8c7

This fingerprint defines dataset version v0.2.

## Relationship to dataset v0.1

Dataset v0.2 supersedes rejected dataset v0.1.

Dataset v0.1 fingerprint:

2486137265534c4bf24b0951877e48957f41f25c71f3c06a083f93b735c1e54f

Dataset v0.1 was not used for model training.

The v0.2 amendment replaces model-facing opaque request composite
tokens with entangled distributed request controls.

The linguistic targets, lexical familiarization, content inventory,
split structure, lexical-transfer holdouts, and semantic experimental
design remain unchanged.

## Trainability audit

Dataset v0.2 passes the token-level training-support requirement.

Every individual model-facing token used by:

- validation;
- IID evaluation;
- compositional-OOD evaluation;
- lexical-transfer OOD evaluation

has training support.

The two complete compositional-OOD request-control combinations remain
absent from register-conditioned training.

Therefore compositional-OOD evaluation tests a novel combination of
familiar control tokens rather than an untrained token embedding.

## Audit status

The GitHub Actions freeze workflow successfully:

1. regenerated dataset v0.2 from source;
2. exported deterministic JSONL files;
3. independently validated the exported records;
4. verified the preregistered split counts;
5. verified request-control integrity;
6. verified absence of opaque request codes from model-facing input;
7. verified complete individual-token training support;
8. verified absence of compositional-OOD combination leakage;
9. verified the unchanged lexical-familiarization artifact;
10. calculated file-level SHA-256 hashes;
11. independently recalculated the complete dataset fingerprint.

The workflow completed successfully before model training began.

## Reproducibility rule

All primary synthetic experiments must use the exact dataset
configuration represented by the v0.2 fingerprint above.

If regeneration produces a different fingerprint, primary training
must not proceed until the discrepancy is understood and documented.
