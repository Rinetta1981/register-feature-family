import pytest

from register_feature_family.generator import (
    build_request_text,
    generate_record,
    infer_power_relation,
)
from register_feature_family.schemas import (
    PowerRelation,
    Regime,
    Split,
    SpeechAct,
)


def test_power_relation_lower_to_higher() -> None:
    relation = infer_power_relation(
        speaker_role="role_01",
        addressee_role="role_03",
    )

    assert relation == PowerRelation.LOWER_TO_HIGHER


def test_request_text_matches_codebook() -> None:
    text = build_request_text(
        content_id="send_report",
        register_code="<C03>",
    )

    assert text == "kelo mira navo daru"


def test_generate_request_record() -> None:
    record = generate_record(
        example_id="request-000001",
        comparison_group_id="send-report-001",
        seed=0,
        regime=Regime.BALANCED,
        split=Split.TRAIN,
        content_id="send_report",
        speaker_role="role_01",
        addressee_role="role_03",
        register_code="<C03>",
    )

    assert record.speech_act == SpeechAct.REQUEST
    assert record.target_text == "kelo mira navo daru"
    assert record.power_relation == PowerRelation.LOWER_TO_HIGHER
    assert record.context_text == (
        "<ROLE_01> <ROLE_03> <CONTENT_01> <C03>"
    )


def test_generate_assertion_record() -> None:
    record = generate_record(
        example_id="assertion-000001",
        comparison_group_id="result-difference-001",
        seed=0,
        regime=Regime.BALANCED,
        split=Split.TRAIN,
        content_id="result_difference",
        speaker_role="role_03",
        addressee_role="role_01",
        register_code="<C11>",
    )

    assert record.speech_act == SpeechAct.ASSERTION
    assert record.target_text == "sava luma caron"
    assert record.power_relation == PowerRelation.HIGHER_TO_LOWER


def test_generation_is_deterministic() -> None:
    kwargs = {
        "example_id": "request-000001",
        "comparison_group_id": "send-report-001",
        "seed": 0,
        "regime": Regime.BALANCED,
        "split": Split.TRAIN,
        "content_id": "send_report",
        "speaker_role": "role_01",
        "addressee_role": "role_03",
        "register_code": "<C03>",
    }

    first = generate_record(**kwargs)
    second = generate_record(**kwargs)

    assert first == second


def test_request_builder_rejects_assertion_code() -> None:
    with pytest.raises(
        ValueError,
        match="<C08> is not a request register code.",
    ):
        build_request_text(
            content_id="send_report",
            register_code="<C08>",
        )
