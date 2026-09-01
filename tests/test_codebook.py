from register_feature_family.codebook import (
    ASSERTION_CONTENT,
    COMPOSITE_REGISTER_CODES,
    REQUEST_CONTENT,
    LexicalPair,
)
from register_feature_family.schemas import (
    Directness,
    EpistemicStance,
    LexicalFormality,
    PolitenessMitigation,
    SpeechAct,
)


def test_lexical_pair_selects_expected_form() -> None:
    pair = LexicalPair(
        less_formal="navo",
        more_formal="terin",
    )

    assert pair.select(LexicalFormality.LESS_FORMAL) == "navo"
    assert pair.select(LexicalFormality.MORE_FORMAL) == "terin"


def test_lexical_pairs_use_distinct_surface_forms() -> None:
    request_pairs = [
        content.verb
        for content in REQUEST_CONTENT.values()
    ]
    assertion_pairs = [
        content.predicate
        for content in ASSERTION_CONTENT.values()
    ]

    for pair in request_pairs + assertion_pairs:
        assert pair.less_formal != pair.more_formal


def test_composite_codebook_contains_twelve_codes() -> None:
    assert len(COMPOSITE_REGISTER_CODES) == 12


def test_request_codes_cover_all_register_combinations() -> None:
    actual = {
        (
            config.lexical_formality,
            config.directness,
            config.politeness_mitigation,
        )
        for config in COMPOSITE_REGISTER_CODES.values()
        if config.speech_act == SpeechAct.REQUEST
    }

    expected = {
        (
            formality,
            directness,
            mitigation,
        )
        for formality in (
            LexicalFormality.LESS_FORMAL,
            LexicalFormality.MORE_FORMAL,
        )
        for directness in (
            Directness.DIRECT,
            Directness.INDIRECT,
        )
        for mitigation in (
            PolitenessMitigation.BARE,
            PolitenessMitigation.MITIGATED,
        )
    }

    assert actual == expected


def test_assertion_codes_cover_all_register_combinations() -> None:
    actual = {
        (
            config.lexical_formality,
            config.epistemic_stance,
        )
        for config in COMPOSITE_REGISTER_CODES.values()
        if config.speech_act == SpeechAct.ASSERTION
    }

    expected = {
        (
            formality,
            stance,
        )
        for formality in (
            LexicalFormality.LESS_FORMAL,
            LexicalFormality.MORE_FORMAL,
        )
        for stance in (
            EpistemicStance.CATEGORICAL,
            EpistemicStance.HEDGED,
        )
    }

    assert actual == expected


def test_request_codes_have_no_epistemic_stance() -> None:
    for config in COMPOSITE_REGISTER_CODES.values():
        if config.speech_act == SpeechAct.REQUEST:
            assert (
                config.epistemic_stance
                == EpistemicStance.NOT_APPLICABLE
            )


def test_assertion_codes_have_no_request_dimensions() -> None:
    for config in COMPOSITE_REGISTER_CODES.values():
        if config.speech_act == SpeechAct.ASSERTION:
            assert config.directness == Directness.NOT_APPLICABLE
            assert (
                config.politeness_mitigation
                == PolitenessMitigation.NOT_APPLICABLE
            )
