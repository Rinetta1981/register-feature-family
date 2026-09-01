import pytest
from pydantic import ValidationError

from register_feature_family.schemas import DatasetRecord


def test_valid_request_is_accepted() -> None:
    record = DatasetRecord(
        example_id="request-000001",
        comparison_group_id="send-report-001",
        seed=0,
        regime="balanced",
        split="train",
        speech_act="request",
        content_id="send_report",
        template_id="request_direct_bare_v1",
        speaker_role="junior_analyst",
        addressee_role="director",
        power_relation="lower_to_higher",
        lexical_formality="less_formal",
        directness="direct",
        politeness_mitigation="bare",
        epistemic_stance="not_applicable",
        context_text="A junior analyst is speaking to a director.",
        target_text="Synthetic request placeholder.",
    )

    assert record.speech_act.value == "request"


def test_request_with_epistemic_stance_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Requests must use epistemic_stance='not_applicable'.",
    ):
        DatasetRecord(
            example_id="request-invalid-001",
            comparison_group_id="send-report-001",
            seed=0,
            regime="balanced",
            split="train",
            speech_act="request",
            content_id="send_report",
            template_id="request_direct_bare_v1",
            speaker_role="junior_analyst",
            addressee_role="director",
            power_relation="lower_to_higher",
            lexical_formality="less_formal",
            directness="direct",
            politeness_mitigation="bare",
            epistemic_stance="hedged",
            context_text="A junior analyst is speaking to a director.",
            target_text="Synthetic request placeholder.",
        )


def test_valid_assertion_is_accepted() -> None:
    record = DatasetRecord(
        example_id="assertion-000001",
        comparison_group_id="result-difference-001",
        seed=0,
        regime="balanced",
        split="train",
        speech_act="assertion",
        content_id="result_difference",
        template_id="assertion_hedged_v1",
        speaker_role="senior_analyst",
        addressee_role="junior_analyst",
        power_relation="higher_to_lower",
        lexical_formality="more_formal",
        directness="not_applicable",
        politeness_mitigation="not_applicable",
        epistemic_stance="hedged",
        context_text="A senior analyst is speaking to a junior analyst.",
        target_text="Synthetic assertion placeholder.",
    )

    assert record.speech_act.value == "assertion"


def test_assertion_with_request_directness_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Assertions must use directness='not_applicable'.",
    ):
        DatasetRecord(
            example_id="assertion-invalid-001",
            comparison_group_id="result-difference-001",
            seed=0,
            regime="balanced",
            split="train",
            speech_act="assertion",
            content_id="result_difference",
            template_id="assertion_hedged_v1",
            speaker_role="senior_analyst",
            addressee_role="junior_analyst",
            power_relation="higher_to_lower",
            lexical_formality="more_formal",
            directness="direct",
            politeness_mitigation="not_applicable",
            epistemic_stance="hedged",
            context_text="A senior analyst is speaking to a junior analyst.",
            target_text="Synthetic assertion placeholder.",
        )
