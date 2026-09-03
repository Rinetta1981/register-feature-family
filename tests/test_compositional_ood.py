from collections import Counter

from register_feature_family.codebook import COMPOSITE_REGISTER_CODES
from register_feature_family.schemas import (
    Directness,
    LexicalFormality,
    PolitenessMitigation,
    SpeechAct,
)
from register_feature_family.split_config import (
    REQUEST_COMPOSITIONAL_OOD_CODES,
    REQUEST_COMPOSITIONAL_OOD_FEATURES,
    is_compositional_ood_register_code,
    training_supported_register_codes,
)


def test_request_compositional_ood_has_two_codes() -> None:
    assert len(REQUEST_COMPOSITIONAL_OOD_CODES) == 2


def test_request_compositional_ood_matches_frozen_features() -> None:
    actual = {
        (
            COMPOSITE_REGISTER_CODES[code].lexical_formality,
            COMPOSITE_REGISTER_CODES[code].directness,
            COMPOSITE_REGISTER_CODES[code].politeness_mitigation,
        )
        for code in REQUEST_COMPOSITIONAL_OOD_CODES
    }

    assert actual == set(REQUEST_COMPOSITIONAL_OOD_FEATURES)


def test_request_training_keeps_six_register_configurations() -> None:
    codes = training_supported_register_codes(SpeechAct.REQUEST)

    assert len(codes) == 6
    assert not set(codes) & set(REQUEST_COMPOSITIONAL_OOD_CODES)


def test_assertion_training_keeps_all_four_configurations() -> None:
    codes = training_supported_register_codes(SpeechAct.ASSERTION)

    assert len(codes) == 4


def test_request_training_marginals_remain_balanced() -> None:
    codes = training_supported_register_codes(SpeechAct.REQUEST)
    configs = [COMPOSITE_REGISTER_CODES[code] for code in codes]

    lexical_counts = Counter(
        config.lexical_formality for config in configs
    )
    directness_counts = Counter(
        config.directness for config in configs
    )
    mitigation_counts = Counter(
        config.politeness_mitigation for config in configs
    )

    assert lexical_counts == {
        LexicalFormality.LESS_FORMAL: 3,
        LexicalFormality.MORE_FORMAL: 3,
    }
    assert directness_counts == {
        Directness.DIRECT: 3,
        Directness.INDIRECT: 3,
    }
    assert mitigation_counts == {
        PolitenessMitigation.BARE: 3,
        PolitenessMitigation.MITIGATED: 3,
    }


def test_request_training_preserves_lexical_directness_pairs() -> None:
    codes = training_supported_register_codes(SpeechAct.REQUEST)
    configs = [COMPOSITE_REGISTER_CODES[code] for code in codes]

    observed = {
        (config.lexical_formality, config.directness)
        for config in configs
    }

    expected = {
        (LexicalFormality.LESS_FORMAL, Directness.DIRECT),
        (LexicalFormality.LESS_FORMAL, Directness.INDIRECT),
        (LexicalFormality.MORE_FORMAL, Directness.DIRECT),
        (LexicalFormality.MORE_FORMAL, Directness.INDIRECT),
    }

    assert observed == expected


def test_request_training_preserves_lexical_mitigation_pairs() -> None:
    codes = training_supported_register_codes(SpeechAct.REQUEST)
    configs = [COMPOSITE_REGISTER_CODES[code] for code in codes]

    observed = {
        (
            config.lexical_formality,
            config.politeness_mitigation,
        )
        for config in configs
    }

    expected = {
        (
            LexicalFormality.LESS_FORMAL,
            PolitenessMitigation.BARE,
        ),
        (
            LexicalFormality.LESS_FORMAL,
            PolitenessMitigation.MITIGATED,
        ),
        (
            LexicalFormality.MORE_FORMAL,
            PolitenessMitigation.BARE,
        ),
        (
            LexicalFormality.MORE_FORMAL,
            PolitenessMitigation.MITIGATED,
        ),
    }

    assert observed == expected


def test_request_training_preserves_directness_mitigation_pairs() -> None:
    codes = training_supported_register_codes(SpeechAct.REQUEST)
    configs = [COMPOSITE_REGISTER_CODES[code] for code in codes]

    observed = {
        (
            config.directness,
            config.politeness_mitigation,
        )
        for config in configs
    }

    expected = {
        (Directness.DIRECT, PolitenessMitigation.BARE),
        (Directness.DIRECT, PolitenessMitigation.MITIGATED),
        (Directness.INDIRECT, PolitenessMitigation.BARE),
        (Directness.INDIRECT, PolitenessMitigation.MITIGATED),
    }

    assert observed == expected


def test_request_holdout_codes_are_classified_as_compositional_ood() -> None:
    for code in REQUEST_COMPOSITIONAL_OOD_CODES:
        assert is_compositional_ood_register_code(
            register_code=code,
            speech_act=SpeechAct.REQUEST,
        )


def test_assertions_have_no_compositional_ood_holdout() -> None:
    for code in training_supported_register_codes(SpeechAct.ASSERTION):
        assert not is_compositional_ood_register_code(
            register_code=code,
            speech_act=SpeechAct.ASSERTION,
        )
