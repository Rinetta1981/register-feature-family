import pytest

from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.schemas import SpeechAct
from register_feature_family.split_config import (
    ASSERTION_LEXICAL_TRANSFER_HOLDOUTS,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
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
