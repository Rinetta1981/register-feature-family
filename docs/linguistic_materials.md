# Controlled Linguistic Materials

## Status

**Draft version 0.1**

This document defines the semantic and social materials that will later
be used by the synthetic data generator.

The underlying meaning of an example is defined separately from the
linguistic form used to express it.

## 1. Design principle

The experiment must distinguish:

1. what is being communicated;
2. who is communicating;
3. to whom it is being communicated;
4. how it is linguistically expressed.

Register variables must not be allowed to change the underlying semantic
content unintentionally.

For this reason, semantic content will be represented using abstract
content identifiers before surface sentences are generated.

## 2. Request content frames

The initial request family will use the following abstract actions.

| Content ID | Underlying requested action |
|---|---|
| `send_report` | Cause the report to be sent to the speaker |
| `review_document` | Examine a document |
| `schedule_meeting` | Arrange a meeting time |
| `update_file` | Modify a file with current information |
| `share_notes` | Provide notes to the speaker |
| `confirm_date` | Verify that a specified date is correct |
| `revise_draft` | Modify a draft |
| `submit_form` | Deliver a completed form |
| `explain_result` | Provide an explanation of a result |
| `correct_record` | Fix inaccurate information in a record |
| `inspect_sample` | Examine a sample |
| `approve_request` | Give approval for a request |

These content IDs describe meaning rather than exact wording.

For example, `send_report` does not yet specify whether the final
utterance will contain words such as:

- send;
- forward;
- please;
- could;
- would.

Those choices belong to the register manipulation.

## 3. Assertion content frames

The initial assertion family will use propositions whose truth conditions
can remain constant while epistemic stance changes.

| Content ID | Underlying proposition |
|---|---|
| `result_difference` | The result shows a difference between conditions |
| `sample_contamination` | The sample is contaminated |
| `meeting_delayed` | The meeting has been delayed |
| `report_complete` | The report is complete |
| `system_unstable` | The system is unstable |
| `estimate_inaccurate` | The estimate is inaccurate |
| `record_incomplete` | The record is incomplete |
| `method_effective` | The method is effective |
| `schedule_conflict` | A scheduling conflict exists |
| `document_outdated` | The document is outdated |
| `measurement_error` | A measurement error occurred |
| `procedure_changed` | The procedure has changed |

The proposition must remain constant when categorical and hedged versions
are compared.

## 4. Social roles

The first study will use organizational roles with relatively clear
power relations.

### Higher-power roles

- `director`
- `department_head`
- `senior_manager`

### Middle or context-dependent roles

- `analyst`
- `researcher`
- `coordinator`

### Lower-power roles

- `junior_analyst`
- `research_assistant`
- `trainee`

The role inventory will be used to construct:

- lower-to-higher interactions;
- equal-power interactions;
- higher-to-lower interactions.

## 5. Power relation

Power relation will be determined from the experimental design rather
than inferred from the wording of the generated sentence.

The permitted values are:

- `lower_to_higher`
- `equal`
- `higher_to_lower`

For example:

```text
speaker_role = junior_analyst
addressee_role = director
power_relation = lower_to_higher


## 12. Core lexical-formality strategy

The primary synthetic experiment will not use natural-English synonyms
as the main manipulation of lexical formality.

Instead, each underlying semantic item will initially receive two
synthetic lexical realizations:

- a lower-register realization;
- a higher-register realization.

The two realizations will have identical meanings by construction.

For example, an abstract action such as `SEND` may eventually receive
two artificial lexical forms that differ only in their assigned
register function.

The actual synthetic vocabulary will be generated systematically rather
than chosen from the illustrative examples in this document.

### Why synthetic lexical pairs are used

Natural-language pairs such as `send` and `forward` may differ in
meaning, collocation, frequency, or other properties in addition to
formality.

Synthetic lexical pairs allow semantic meaning to remain exactly
constant while lexical register varies.

This provides ground truth for the mechanistic analysis.

### Natural-language replication

Natural-English lexical contrasts will be introduced only in the later
pretrained-model replication.

This will allow the project to separate:

1. a high-control synthetic mechanistic study;
2. a lower-control but more ecologically valid natural-language study.

## 13. Conditional-generation requirement

Every training example must contain sufficient information for the
model to determine the intended register realization.

The same semantic and social context must not map unpredictably to
multiple incompatible target forms.

The toy-language generator will therefore include controlled latent
cues specifying the intended register configuration.

These cues will use neutral synthetic symbols rather than natural
labels such as `formal`, `polite`, or `indirect`.

This prevents the experiment from giving the model the linguistic
interpretation directly.

The exact cue system will be defined before implementation.

## 14. Decisions remaining before implementation

The following must still be defined:

- synthetic vocabulary construction;
- neutral control-cue system;
- direct-request templates;
- indirect-request templates;
- mitigation strategies;
- categorical assertion templates;
- hedged assertion templates;
- train and test template allocation;
- dataset sizes;
- lexical OOD design.
