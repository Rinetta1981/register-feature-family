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


## 3. Speech-act codes

The initial speech-act codes are:

* `<REQ>` for requests;
* `<AST>` for assertions.

These codes identify the grammatical family but do not encode register.

## 4. Semantic-content codes

Each underlying action or proposition will receive a neutral content
identifier.

Examples:

```text
<CONTENT_01>
<CONTENT_02>
<CONTENT_03>
```

The identifier will not contain words such as `send`, `report`, or
`meeting`.

The mapping between content codes and meanings will be stored in the
generator configuration rather than exposed in the token name.

## 5. Lexical-formality control

For the factorized calibration condition, lexical register will use two
neutral codes:

```text
<LX0>
<LX1>
```

One code will correspond to the lower-register lexical realization and
the other to the higher-register realization.

The code names themselves do not reveal which is which.

The mapping will be fixed before training and stored in the experiment
configuration.

## 6. Directness control

For the factorized calibration condition, request directness will use:

```text
<SY0>
<SY1>
```

One code corresponds to a direct request structure.

The other corresponds to an indirect request structure.

For assertions, this field will use:

```text
<SY_NA>
```

## 7. Mitigation control

For the factorized calibration condition, request mitigation will use:

```text
<MT0>
<MT1>
```

One code corresponds to a bare request.

The other corresponds to a mitigated request.

For assertions, this field will use:

```text
<MT_NA>
```

## 8. Epistemic-stance control

For the factorized calibration condition, assertions will use:

```text
<EP0>
<EP1>
```

One code corresponds to categorical stance.

The other corresponds to hedged stance.

For requests, this field will use:

```text
<EP_NA>
```

## 9. Power relation and participant roles

Speaker and addressee roles will be represented using neutral role
codes.

For example:

```text
<SPK_03>
<ADR_08>
```

The power relation will be stored as ground-truth metadata but will not
normally be supplied as a separate input token in the primary
experiment.

Instead, the relation will be determined by the combination of speaker
and addressee roles.

For example, the experiment configuration may specify that one role is
higher in institutional power than another.

### Why power is not explicitly cued

Providing both participant roles and a separate power token would give
the model redundant information.

Removing the explicit power token allows later analyses to test whether
the model derives a representation of relative power from participant
identity and context.

A separate explicit-power condition may be used later as a diagnostic
control.

## 10. Synthetic lexical vocabulary

Each semantic lexical item will receive at least two artificial surface
forms with identical meaning by construction.

Illustrative example only:

| Meaning | Lower-register form | Higher-register form |
| ------- | ------------------- | -------------------- |
| SEND    | `navo`              | `terin`              |
| REVIEW  | `peka`              | `solim`              |
| REPORT  | `daru`              | `velan`              |

The final vocabulary will be generated systematically rather than
manually chosen from these examples.

## 11. Vocabulary-construction rules

Synthetic lexical forms must:

1. be pronounceable enough to inspect manually;
2. avoid existing common English words;
3. avoid obvious semantic associations;
4. have similar length distributions across register classes;
5. avoid one class having systematically distinctive prefixes or
   suffixes;
6. avoid accidental overlap between forms;
7. be deterministically reproducible from a fixed seed.

The generator will verify these properties where possible.

## 12. Request grammar

The initial request grammar will support four combinations:

```text
direct + bare
direct + mitigated
indirect + bare
indirect + mitigated
```

Conceptually:

```text
DIRECT + BARE:
VERB OBJECT

DIRECT + MITIGATED:
MITIGATION VERB OBJECT

INDIRECT + BARE:
INDIRECT_MARKER VERB OBJECT

INDIRECT + MITIGATED:
INDIRECT_MARKER MITIGATION VERB OBJECT
```

The synthetic grammar will use controlled markers rather than relying
only on English constructions.

This allows directness and mitigation to vary independently.

## 13. Assertion grammar

Assertions will initially vary:

* lexical register;
* epistemic stance.

Conceptually:

```text
CATEGORICAL:
SUBJECT PREDICATE

HEDGED:
HEDGE SUBJECT PREDICATE
```

The hedge marker will alter expressed certainty without changing the
underlying proposition.

## 14. Not-applicable dimensions

A dimension that does not apply to a speech-act family will receive an
explicit neutral `NA` code in the factorized calibration condition.

Examples:

```text
Request:
<EP_NA>

Assertion:
<SY_NA> <MT_NA>
```

This keeps the calibration input structure consistent across examples.

## 15. Avoiding cue leakage

Control codes must not contain readable semantic labels.

For example, the following are prohibited:

```text
<FORMAL>
<POLITE>
<INDIRECT>
<HEDGED>
```

The following style is permitted:

```text
<LX0>
<MT1>
<SY0>
<EP1>
```

The interpretation of each code will be stored separately.

## 16. Control-code robustness check

A robustness experiment will repeat the main analysis after permuting
the mapping between arbitrary code values and linguistic realizations.

For example, if one code originally maps to the lower-register lexical
form, a replication may reverse that mapping.

The scientific conclusion should not depend on arbitrary token names.

## 17. Primary and calibration cue designs

The experiment will contain two cue designs.

### 17.1 Primary design: composite register codes

The primary mechanistic experiment will use a single arbitrary code to
represent the complete intended register configuration.

For example:

```text
<C17>
```

may correspond internally to a configuration such as:

* higher-register lexical realization;
* indirect syntax;
* mitigated request.

The model will not receive separate cues identifying these component
dimensions.

The mapping between composite codes and register configurations will be
stored in the experiment configuration and will not be visible in the
token names.

### Why this is the primary design

If separate input tokens are provided for lexical register, directness,
mitigation, and stance, the input structure itself may encourage the
model to maintain separate internal representations.

Using one composite code makes the mechanistic question harder:

Can the model develop separable representations of register dimensions
even when the input does not explicitly factorize them?

### 17.2 Calibration design: factorized register codes

A secondary calibration condition will use separate neutral cues such
as:

```text
<LX0>
<SY1>
<MT0>
<EP1>
```

This condition provides a simpler setting in which the intended
dimensions are explicitly factorized.

Its purpose is to verify that the probing, sparse-feature, and causal
intervention methods can recover known separable structure.

Results from this calibration condition will not by themselves be
treated as evidence that register naturally decomposes into independent
features.

### 17.3 Comparison

The main analysis will compare:

1. factor recovery in the factorized calibration condition;
2. factor recovery in the composite-code primary condition;
3. causal selectivity of recovered representations in both conditions.

Evidence for a feature-family account will be strongest if separable
causal features emerge in the composite-code condition.

## 18. Example conceptual records

### Request

```text
<REQ>
<CONTENT_01>
<SPK_03>
<ADR_08>
<C17>
```

Target:

```text
synthetic indirect mitigated request
```

### Assertion

```text
<AST>
<CONTENT_07>
<SPK_06>
<ADR_02>
<C05>
```

Target:

```text
synthetic hedged assertion
```

These are structural illustrations, not final training examples.

## 19. Decisions remaining before implementation

Before Python implementation begins, the following must still be fixed:

* number of semantic content items;
* number of synthetic lexical items;
* phonological form-generation algorithm;
* exact request markers;
* exact assertion markers;
* role-code mapping;
* composite-code mapping;
* factorized calibration mapping;
* train and test vocabulary allocation;
* dataset sizes.

