from register_feature_family.content_inventory import (
    ALL_CONTENT_IDS,
    ASSERTION_CONTENT_IDS,
    CONTENT_CODE_BY_ID,
    REQUEST_CONTENT_IDS,
)


def test_content_inventory_has_expected_size() -> None:
    assert len(REQUEST_CONTENT_IDS) == 16
    assert len(ASSERTION_CONTENT_IDS) == 16
    assert len(ALL_CONTENT_IDS) == 32


def test_content_ids_are_globally_unique() -> None:
    assert len(set(ALL_CONTENT_IDS)) == 32


def test_every_content_has_unique_neutral_code() -> None:
    codes = list(CONTENT_CODE_BY_ID.values())

    assert len(codes) == 32
    assert len(set(codes)) == 32


def test_content_code_boundaries() -> None:
    assert CONTENT_CODE_BY_ID["send_report"] == "<CONTENT_01>"
    assert CONTENT_CODE_BY_ID["verify_entry"] == "<CONTENT_16>"
    assert CONTENT_CODE_BY_ID["result_difference"] == "<CONTENT_17>"
    assert CONTENT_CODE_BY_ID["timeline_changed"] == "<CONTENT_32>"
