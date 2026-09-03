# Semantic Content Inventory

## Status

**Proposed freeze version 0.1**

This inventory is defined before model training or inspection of
experimental results.

It becomes frozen when merged into `main`.

Any later additions, removals, or substitutions must be documented as
an amendment.

## 1. Purpose

The full synthetic study will contain:

* 16 request content items;
* 16 assertion content items.

The natural-language descriptions in this document are for researchers
only.

The toy model will not receive labels such as `send_report` or natural-
language glosses such as "send the report."

Model inputs will instead use neutral identifiers such as:

```text
<CONTENT_01>
<CONTENT_02>
```

## 2. Linguistic architecture

Each request content item will contain:

```text
REGISTER-SENSITIVE VERB + FIXED OBJECT
```

The verb will have two synthetic lexical realizations with identical
meaning by construction.

The object will remain fixed across lexical-register conditions.

Each assertion content item will contain:

```text
FIXED SUBJECT + REGISTER-SENSITIVE PREDICATE
```

The predicate will have two synthetic lexical realizations with
identical meaning by construction.

The subject will remain fixed across lexical-register conditions.

This localizes the lexical-register manipulation to one lexical element
per utterance.

## 3. Request inventory

| Content ID         | Researcher gloss     |
| ------------------ | -------------------- |
| `send_report`      | send the report      |
| `review_document`  | review the document  |
| `schedule_meeting` | schedule the meeting |
| `update_file`      | update the file      |
| `share_notes`      | share the notes      |
| `confirm_date`     | confirm the date     |
| `revise_draft`     | revise the draft     |
| `submit_form`      | submit the form      |
| `explain_result`   | explain the result   |
| `correct_record`   | correct the record   |
| `inspect_sample`   | inspect the sample   |
| `approve_request`  | approve the request  |
| `archive_message`  | archive the message  |
| `compare_versions` | compare the versions |
| `annotate_chart`   | annotate the chart   |
| `verify_entry`     | verify the entry     |

All sixteen request meanings must remain compatible with the same
abstract request grammar.

Differences in English wording are not used as experimental variables in
the toy model.

## 4. Assertion inventory

| Content ID            | Researcher gloss                      |
| --------------------- | ------------------------------------- |
| `result_difference`   | the result differs between conditions |
| `system_unstable`     | the system is unstable                |
| `sample_contaminated` | the sample is contaminated            |
| `meeting_delayed`     | the meeting is delayed                |
| `report_complete`     | the report is complete                |
| `estimate_inaccurate` | the estimate is inaccurate            |
| `record_incomplete`   | the record is incomplete              |
| `method_effective`    | the method is effective               |
| `schedule_conflict`   | a scheduling conflict exists          |
| `document_outdated`   | the document is outdated              |
| `measurement_error`   | a measurement error occurred          |
| `procedure_changed`   | the procedure changed                 |
| `sample_ready`        | the sample is ready                   |
| `model_consistent`    | the model is consistent               |
| `request_valid`       | the request is valid                  |
| `timeline_changed`    | the timeline changed                  |

All sixteen assertion meanings will use the same abstract
subject–predicate grammar.

The synthetic grammar does not require the natural-language glosses to
share the same English syntactic category.

## 5. Content-code assignment

Neutral model-facing content codes will be assigned deterministically.

Request contents will receive:

```text
<CONTENT_01>
...
<CONTENT_16>
```

Assertion contents will receive:

```text
<CONTENT_17>
...
<CONTENT_32>
```

The mapping will be stored in configuration and tested automatically.

The model-facing code names will not contain semantic information.

## 6. Lexical-transfer selection

No content item is manually designated as lexical OOD in this document.

After this inventory is frozen, the lexical-transfer items will be
selected algorithmically using the predetermined split seed:

```text
2026
```

Selection will occur separately for requests and assertions.

For each speech-act family:

* 12 content items will participate in register-conditioned training;
* 4 content items will be reserved from register-conditioned training
  for lexical-transfer evaluation.

The four held-out items will still appear during neutral vocabulary
familiarization as specified in the dataset-split design.

## 7. Freeze rule

After this inventory is merged into `main`, content items must not be
added, removed, renamed, or substituted because of model performance.

If a content item later proves technically unusable, the reason must be
documented before replacement, and the change must be recorded as a
protocol amendment.
