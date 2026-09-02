# Dataset Split Design

## Status

Draft version 0.1

## 1. Purpose

The dataset will use structure-aware splits rather than random
row-level splitting.

The goal is to distinguish ordinary generalization from compositional
and lexical out-of-distribution generalization.

## 2. Training split

The training split will contain the examples used to update model
parameters.

It will exclude all content reserved for lexical OOD evaluation and all
register combinations specifically reserved for compositional OOD
evaluation.

## 3. Validation split

The validation split will contain unseen examples drawn from structures
permitted in training.

It will be used for development decisions and model selection.

It will not be treated as final test evidence.

## 4. IID test split

The IID test split will contain unseen examples whose component
structures and register combinations are represented during training.

It tests ordinary generalization without introducing a new structural
combination.

## 5. Compositional OOD test split

The compositional OOD test split will contain familiar individual
features combined in configurations withheld from training.

Every component feature value must appear somewhere in training.

The complete held-out combination must not appear in training.

## 6. Lexical OOD test split

The lexical OOD test split will contain semantic content items and their
synthetic lexical realizations that are completely absent from training.

The register system itself remains familiar.

This tests whether the model can apply learned register distinctions to
new lexical content.

## 7. Leakage rules

The implementation must prevent:

1. lexical-OOD content IDs from appearing in training;
2. lexical-OOD surface forms from appearing in training;
3. compositional-OOD register combinations from appearing in training;
4. exact example IDs from appearing in more than one split;
5. identical target strings from crossing prohibited split boundaries
   when they originate from the same content and condition.

## 8. Determinism

Split assignment must be reproducible from a fixed configuration and
seed.

The same configuration must always produce exactly the same split
membership.

## 9. Next decisions

Before implementation, the project must define:

- which semantic items are reserved for lexical OOD;
- which request register combinations are held out compositionally;
- which assertion register combinations are held out compositionally;
- validation and IID allocation rules;
- final split-size targets.
