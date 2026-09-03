# Lexical-Transfer Familiarization

## Status

Protocol amendment version 0.1.

This document clarifies the lexical-transfer familiarization procedure
before model training begins.

## 1. Purpose

The lexical-transfer evaluation asks whether the model can apply a
register system learned on some semantic items to other semantic items
that were not used in register-conditioned training.

The evaluation must distinguish:

1. knowing the lexical material;
2. knowing how that material participates in the register system.

The model must therefore know the surface forms belonging to a held-out
semantic item without having seen that item under register-conditioned
generation.

## 2. Three forms per content item

Every request content contains:

- two register-sensitive verb forms;
- one fixed object form.

Every assertion content contains:

- one fixed subject form;
- two register-sensitive predicate forms.

Therefore each of the 32 semantic contents is associated with exactly
three synthetic surface forms.

All three forms must be available during neutral lexical
familiarization.

Otherwise the fixed object or subject of a lexical-transfer item would
remain completely unseen, making register-conditioned generation
partly impossible for reasons unrelated to register transfer.

## 3. Neutral familiarization interface

Familiarization examples use neutral control symbols rather than
register labels.

Conceptually, each content item receives examples of the form:

```text
<LEX> <CONTENT_xx> <VAR0>  -> register-sensitive form A
<LEX> <CONTENT_xx> <VAR1>  -> register-sensitive form B
<LEX> <CONTENT_xx> <FIXED> -> fixed content form
