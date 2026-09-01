import pytest

from register_feature_family.vocabulary import (
    generate_lexical_pairs,
    generate_unique_forms,
)


def test_unique_forms_are_deterministic() -> None:
    first = generate_unique_forms(count=20, seed=42)
    second = generate_unique_forms(count=20, seed=42)

    assert first == second


def test_unique_forms_are_unique() -> None:
    forms = generate_unique_forms(count=50, seed=42)

    assert len(forms) == len(set(forms))


def test_all_forms_are_cvcv_length() -> None:
    forms = generate_unique_forms(count=50, seed=42)

    assert all(len(form) == 4 for form in forms)


def test_lexical_pairs_are_deterministic() -> None:
    semantic_ids = [
        "send",
        "review",
        "approve",
        "revise",
    ]

    first = generate_lexical_pairs(
        semantic_ids=semantic_ids,
        seed=42,
    )
    second = generate_lexical_pairs(
        semantic_ids=semantic_ids,
        seed=42,
    )

    assert first == second


def test_each_semantic_item_gets_two_distinct_forms() -> None:
    pairs = generate_lexical_pairs(
        semantic_ids=[
            "send",
            "review",
            "approve",
            "revise",
        ],
        seed=42,
    )

    for pair in pairs:
        assert pair.less_formal != pair.more_formal


def test_all_pair_forms_are_globally_unique() -> None:
    pairs = generate_lexical_pairs(
        semantic_ids=[
            "send",
            "review",
            "approve",
            "revise",
        ],
        seed=42,
    )

    forms = [
        form
        for pair in pairs
        for form in (
            pair.less_formal,
            pair.more_formal,
        )
    ]

    assert len(forms) == len(set(forms))


def test_register_classes_have_equal_word_lengths() -> None:
    pairs = generate_lexical_pairs(
        semantic_ids=[
            "send",
            "review",
            "approve",
            "revise",
            "inspect",
            "explain",
        ],
        seed=42,
    )

    less_formal_lengths = [
        len(pair.less_formal)
        for pair in pairs
    ]
    more_formal_lengths = [
        len(pair.more_formal)
        for pair in pairs
    ]

    assert less_formal_lengths == more_formal_lengths


def test_empty_semantic_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_ids must not be empty.",
    ):
        generate_lexical_pairs(
            semantic_ids=[],
            seed=42,
        )


def test_duplicate_semantic_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_ids must be unique.",
    ):
        generate_lexical_pairs(
            semantic_ids=["send", "send"],
            seed=42,
        )
