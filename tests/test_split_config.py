import pytest

from collections import Counter

from register_feature_family.codebook import ROLE_CODES

from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)

from register_feature_family.schemas import SpeechAct, Split
from register_feature_family.split_config import (
    ASSERTION_LEXICAL_TRANSFER_HOLDOUTS,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
    in_distribution_content_ids,
    ordinary_split_for_group,
    select_lexical_transfer_holdouts,
)

def test_request_holdouts_are_frozen() -> None:
    assert REQUEST_LEXICAL_TRANSFER_HOLDOUTS == (
        "confirm_date",
        "explain_result",
        "revise_draft",
        "compare_versions",
    )


def test_assertion_holdouts_are_frozen() -> None:
    assert ASSERTION_LEXICAL_TRANSFER_HOLDOUTS == (
        "document_outdated",
        "sample_contaminated",
        "sample_ready",
        "request_valid",
    )


def test_holdouts_have_expected_size() -> None:
    assert len(REQUEST_LEXICAL_TRANSFER_HOLDOUTS) == 4
    assert len(ASSERTION_LEXICAL_TRANSFER_HOLDOUTS) == 4


def test_selection_is_deterministic() -> None:
    first = select_lexical_transfer_holdouts(
        content_ids=REQUEST_CONTENT_IDS,
        speech_act=SpeechAct.REQUEST,
    )
    second = select_lexical_transfer_holdouts(
        content_ids=REQUEST_CONTENT_IDS,
        speech_act=SpeechAct.REQUEST,
    )

    assert first == second


def test_different_seed_changes_selection() -> None:
    default = select_lexical_transfer_holdouts(
        content_ids=ASSERTION_CONTENT_IDS,
        speech_act=SpeechAct.ASSERTION,
    )
    alternative = select_lexical_transfer_holdouts(
        content_ids=ASSERTION_CONTENT_IDS,
        speech_act=SpeechAct.ASSERTION,
        seed=999,
    )

    assert default != alternative


def test_duplicate_content_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        select_lexical_transfer_holdouts(
            content_ids=("item_a", "item_a", "item_b"),
            speech_act=SpeechAct.REQUEST,
        )


def test_invalid_holdout_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_lexical_transfer_holdouts(
            content_ids=REQUEST_CONTENT_IDS,
            speech_act=SpeechAct.REQUEST,
            count=16,
        )

def test_each_family_has_twelve_in_distribution_contents() -> None:
    request_ids = in_distribution_content_ids(SpeechAct.REQUEST)
    assertion_ids = in_distribution_content_ids(SpeechAct.ASSERTION)

    assert len(request_ids) == 12
    assert len(assertion_ids) == 12


@pytest.mark.parametrize(
    "speech_act",
    [SpeechAct.REQUEST, SpeechAct.ASSERTION],
)
def test_each_content_gets_seven_train_one_validation_one_iid(
    speech_act: SpeechAct,
) -> None:
    content_ids = in_distribution_content_ids(speech_act)
    roles = sorted(ROLE_CODES)

    for content_id in content_ids:
        assignments = [
            ordinary_split_for_group(
                content_id=content_id,
                speech_act=speech_act,
                speaker_role=speaker_role,
                addressee_role=addressee_role,
            )
            for speaker_role in roles
            for addressee_role in roles
        ]

        counts = Counter(assignments)

        assert counts == {
            Split.TRAIN: 7,
            Split.VALIDATION: 1,
            Split.IID_TEST: 1,
        }


@pytest.mark.parametrize(
    "speech_act",
    [SpeechAct.REQUEST, SpeechAct.ASSERTION],
)
def test_family_level_group_counts_are_correct(
    speech_act: SpeechAct,
) -> None:
    content_ids = in_distribution_content_ids(speech_act)
    roles = sorted(ROLE_CODES)

    assignments = [
        ordinary_split_for_group(
            content_id=content_id,
            speech_act=speech_act,
            speaker_role=speaker_role,
            addressee_role=addressee_role,
        )
        for content_id in content_ids
        for speaker_role in roles
        for addressee_role in roles
    ]

    counts = Counter(assignments)

    assert counts == {
        Split.TRAIN: 84,
        Split.VALIDATION: 12,
        Split.IID_TEST: 12,
    }


def test_request_lexical_holdout_cannot_receive_ordinary_split() -> None:
    content_id = REQUEST_LEXICAL_TRANSFER_HOLDOUTS[0]

    with pytest.raises(ValueError):
        ordinary_split_for_group(
            content_id=content_id,
            speech_act=SpeechAct.REQUEST,
            speaker_role="role_01",
            addressee_role="role_02",
        )


def test_assertion_lexical_holdout_cannot_receive_ordinary_split() -> None:
    content_id = ASSERTION_LEXICAL_TRANSFER_HOLDOUTS[0]

    with pytest.raises(ValueError):
        ordinary_split_for_group(
            content_id=content_id,
            speech_act=SpeechAct.ASSERTION,
            speaker_role="role_01",
            addressee_role="role_02",
        )


def test_content_cannot_be_assigned_to_wrong_speech_act() -> None:
    with pytest.raises(ValueError):
        ordinary_split_for_group(
            content_id="send_report",
            speech_act=SpeechAct.ASSERTION,
            speaker_role="role_01",
            addressee_role="role_02",
        )


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError):
        ordinary_split_for_group(
            content_id="send_report",
            speech_act=SpeechAct.REQUEST,
            speaker_role="unknown_role",
            addressee_role="role_02",
        )


def test_ordinary_assignment_is_deterministic() -> None:
    first = ordinary_split_for_group(
        content_id="send_report",
        speech_act=SpeechAct.REQUEST,
        speaker_role="role_01",
        addressee_role="role_03",
    )

    second = ordinary_split_for_group(
        content_id="send_report",
        speech_act=SpeechAct.REQUEST,
        speaker_role="role_01",
        addressee_role="role_03",
    )

    assert first == second
