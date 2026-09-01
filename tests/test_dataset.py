from collections import Counter

from register_feature_family.dataset import (
    generate_balanced_pilot,
    register_codes_for,
    role_pairs,
)
from register_feature_family.schemas import (
    SpeechAct,
)


def test_role_pairs_cover_all_ordered_combinations() -> None:
    pairs = role_pairs()

    assert len(pairs) == 9
    assert len(set(pairs)) == 9


def test_request_register_codes_count() -> None:
    codes = register_codes_for(SpeechAct.REQUEST)

    assert len(codes) == 8


def test_assertion_register_codes_count() -> None:
    codes = register_codes_for(SpeechAct.ASSERTION)

    assert len(codes) == 4


def test_balanced_pilot_has_expected_size() -> None:
    records = generate_balanced_pilot(seed=0)

    assert len(records) == 216


def test_balanced_pilot_has_unique_example_ids() -> None:
    records = generate_balanced_pilot(seed=0)

    example_ids = [record.example_id for record in records]

    assert len(example_ids) == len(set(example_ids))


def test_request_register_configurations_are_balanced() -> None:
    records = generate_balanced_pilot(seed=0)

    request_records = [
        record
        for record in records
        if record.speech_act == SpeechAct.REQUEST
    ]

    counts = Counter(
        (
            record.lexical_formality,
            record.directness,
            record.politeness_mitigation,
        )
        for record in request_records
    )

    assert len(counts) == 8
    assert set(counts.values()) == {18}


def test_assertion_register_configurations_are_balanced() -> None:
    records = generate_balanced_pilot(seed=0)

    assertion_records = [
        record
        for record in records
        if record.speech_act == SpeechAct.ASSERTION
    ]

    counts = Counter(
        (
            record.lexical_formality,
            record.epistemic_stance,
        )
        for record in assertion_records
    )

    assert len(counts) == 4
    assert set(counts.values()) == {18}


def test_generation_is_deterministic() -> None:
    first = generate_balanced_pilot(seed=0)
    second = generate_balanced_pilot(seed=0)

    assert first == second
