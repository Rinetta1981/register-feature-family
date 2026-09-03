import pytest

from register_feature_family.request_controls import (
    REQUEST_COMPOSITE_CODES,
    REQUEST_CONTROL_BY_COMPOSITE_CODE,
    REQUEST_CONTROL_TOKENS,
    request_control_tokens,
)


def test_request_control_mapping_matches_frozen_xor_design() -> None:
    feature_bits = {
        "<C00>": (0, 0, 0),
        "<C01>": (0, 0, 1),
        "<C02>": (0, 1, 0),
        "<C03>": (0, 1, 1),
        "<C04>": (1, 0, 0),
        "<C05>": (1, 0, 1),
        "<C06>": (1, 1, 0),
        "<C07>": (1, 1, 1),
    }

    for code, (
        lexical,
        directness,
        mitigation,
    ) in feature_bits.items():
        expected = (
            f"<RC1_{lexical ^ directness}>",
            f"<RC2_{directness ^ mitigation}>",
            f"<RC3_{lexical ^ directness ^ mitigation}>",
        )

        assert request_control_tokens(code) == expected


def test_request_control_mapping_is_one_to_one() -> None:
    combinations = set(
        REQUEST_CONTROL_BY_COMPOSITE_CODE.values()
    )

    assert len(combinations) == 8


def test_every_control_token_has_non_ood_support() -> None:
    training_codes = {
        "<C00>",
        "<C01>",
        "<C02>",
        "<C05>",
        "<C06>",
        "<C07>",
    }

    supported_tokens: set[str] = set()

    for code in training_codes:
        supported_tokens.update(
            request_control_tokens(code)
        )

    assert supported_tokens == set(
        REQUEST_CONTROL_TOKENS
    )


def test_compositional_ood_controls_are_distinct() -> None:
    assert (
        request_control_tokens("<C03>")
        != request_control_tokens("<C04>")
    )


def test_all_request_codes_are_mapped() -> None:
    assert set(
        REQUEST_CONTROL_BY_COMPOSITE_CODE
    ) == set(REQUEST_COMPOSITE_CODES)


def test_assertion_code_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="not a request composite code",
    ):
        request_control_tokens("<C08>")
