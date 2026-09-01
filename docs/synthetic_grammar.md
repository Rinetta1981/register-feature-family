# Synthetic Grammar and Control Codebook

## Status

**Draft version 0.1**

This document defines the neutral control codes and synthetic vocabulary
used in the toy-language experiment.

No model training will begin until this specification is implemented
and validated with automated tests.

## 1. Purpose

The toy model must receive enough information to determine which
linguistic realization it should generate.

However, the model will not receive natural-language labels such as:

- formal;
- informal;
- polite;
- direct;
- hedged.

Instead, the experiment will use arbitrary control symbols.

Their sociolinguistic interpretation is known to the researcher but is
not encoded in their names.

## 2. General input structure

A training example will conceptually contain:

1. a speech-act code;
2. a semantic-content code;
3. a speaker-role code;
4. an addressee-role code;
5. neutral register-control codes;
6. a target utterance.

Illustrative structure:

```text
<REQ> <CONTENT_03> <SPK_02> <ADR_07> <LX1> <SY0> <MT1>
→ target utterance
