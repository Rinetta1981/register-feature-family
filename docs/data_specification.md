# Synthetic Data Specification

## Status

**Draft version 0.1**

This specification is being written before implementation of the data
generator.

## 1. Purpose

This document defines the structure of the synthetic-language dataset
for the project *Register Is a Feature Family, Not a Dial*.

Each example will record:

- what the speaker is communicating;
- who is speaking;
- who is being addressed;
- the relationship between the participants;
- the register features used;
- the generated utterance;
- the dataset split and experimental condition.

## 2. Unit of analysis

One dataset record represents:

1. one speaker;
2. one addressee;
3. one social context;
4. one proposition or requested action;
5. one target utterance;
6. explicit labels for its register properties.

The dataset will use JSON Lines format, abbreviated as JSONL.

In a JSONL file, each line contains one complete example.

## 3. Speech-act families

The first study will contain two speech-act families.

### 3.1 Requests

Request examples will vary:

- lexical formality;
- syntactic directness;
- politeness mitigation;
- speaker–addressee power relation.

Epistemic stance will be marked `not_applicable`.

Example:

> Could you please review the report?

### 3.2 Assertions

Assertion examples will vary:

- lexical formality;
- epistemic stance;
- speaker–addressee power relation.

Directness and politeness mitigation will be marked `not_applicable`.

Example:

> The findings may indicate a difference.

### 3.3 Why the families are separated

Directness and politeness mitigation are most clearly defined for
requests.

Epistemic stance is most clearly defined for assertions.

We will not force a linguistic variable into a speech act where it does
not have a clear interpretation.

## 4. Required fields

| Field | Meaning |
|---|---|
| `schema_version` | Version of the data specification |
| `example_id` | Unique identifier for one example |
| `comparison_group_id` | Identifier linking controlled variations of the same content |
| `seed` | Random seed used during generation |
| `regime` | Balanced, correlated, or confounded training condition |
| `split` | Dataset partition |
| `speech_act` | Request or assertion |
| `content_id` | Abstract action or proposition |
| `template_id` | Sentence template used |
| `speaker_role` | Role of the speaker |
| `addressee_role` | Role of the addressee |
| `power_relation` | Relative power of the participants |
| `lexical_formality` | Less-formal or more-formal vocabulary |
| `directness` | Direct, indirect, or not applicable |
| `politeness_mitigation` | Bare, mitigated, or not applicable |
| `epistemic_stance` | Categorical, hedged, or not applicable |
| `context_text` | Description of the social context |
| `target_text` | Utterance generated for the example |

## 5. Permitted labels

### `regime`

- `balanced`
- `correlated`
- `confounded`

### `split`

- `train`
- `validation`
- `iid_test`
- `compositional_ood_test`
- `lexical_ood_test`

### `speech_act`

- `request`
- `assertion`

### `power_relation`

- `lower_to_higher`
- `equal`
- `higher_to_lower`

### `lexical_formality`

- `less_formal`
- `more_formal`

### `directness`

- `direct`
- `indirect`
- `not_applicable`

### `politeness_mitigation`

- `bare`
- `mitigated`
- `not_applicable`

### `epistemic_stance`

- `categorical`
- `hedged`
- `not_applicable`

Alternative spellings or abbreviations will not be permitted.

For example, the dataset must consistently use `more_formal`, rather
than mixing `more_formal`, `formal`, and `high`.

## 6. Example request record

```json
{
  "schema_version": "0.1",
  "example_id": "request-000001",
  "comparison_group_id": "send-report-001",
  "seed": 0,
  "regime": "balanced",
  "split": "train",
  "speech_act": "request",
  "content_id": "send_report",
  "template_id": "request_indirect_mitigated_v1",
  "speaker_role": "junior_analyst",
  "addressee_role": "director",
  "power_relation": "lower_to_higher",
  "lexical_formality": "more_formal",
  "directness": "indirect",
  "politeness_mitigation": "mitigated",
  "epistemic_stance": "not_applicable",
  "context_text": "A junior analyst is speaking to a director.",
  "target_text": "Could you please forward the report?"
}

{
  "schema_version": "0.1",
  "example_id": "assertion-000001",
  "comparison_group_id": "result-difference-001",
  "seed": 0,
  "regime": "balanced",
  "split": "train",
  "speech_act": "assertion",
  "content_id": "result_difference",
  "template_id": "assertion_hedged_v1",
  "speaker_role": "senior_analyst",
  "addressee_role": "junior_analyst",
  "power_relation": "higher_to_lower",
  "lexical_formality": "more_formal",
  "directness": "not_applicable",
  "politeness_mitigation": "not_applicable",
  "epistemic_stance": "hedged",
  "context_text": "A senior analyst is speaking to a junior analyst.",
  "target_text": "The findings may indicate a difference."
}

