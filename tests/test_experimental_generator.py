from collections import Counter

from register_feature_family.codebook import (
    COMPOSITE_REGISTER_CODES,
    HEDGE_MARKER,
    INDIRECT_MARKER,
    MITIGATION_MARKER,
    ROLE_CODES,
)
from register_feature_family.content_inventory import CONTENT_CODE_BY_ID
from register_feature_family.experimental_codebook import (
    EXPERIMENTAL_ASSERTION_CONTENT,
    EXPERIMENTAL_REQUEST_CONTENT,
    experimental_surface_forms,
)
from register_feature_family.experimental_generator import (
    FamiliarizationVariant,
    LEXICAL_FAMILIARIZATION_CODE,
    generate_confirmatory_records,
    generate_experimental_record,
    generate_lexical_familiarization,
)
from register_feature_family.schemas import (
    Directness,
    EpistemicStance,
    PolitenessMitigation,
    SpeechAct,
    Split,
)
from register_feature_family.split_config import (
    ASSERTION_LEXICAL_TRANSFER_HOLDOUTS,
    REQUEST_COMPOSITIONAL_OOD_CODES,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
)


def test_confirmatory_generator_has_expected_size() -> None:
    records = generate_confirmatory_records(seed=0)

    assert len(records) == 1656


def test_confirmatory_split_counts_match_protocol() -> None:
    records = generate_confirmatory_records(seed=0)
    counts = Counter(record.split for record in records)

    assert counts == {
        Split.TRAIN: 840,
        Split.VALIDATION: 120,
        Split.IID_TEST: 120,
        Split.COMPOSITIONAL_OOD_TEST: 216,
        Split.LEXICAL_OOD_TEST: 360,
    }


def test_confirmatory_example_ids_are_globally_unique() -> None:
    records = generate_confirmatory_records(seed=0)
    example_ids = [record.example_id for record in records]

    assert len(example_ids) == len(set(example_ids))


def test_contexts_use_neutral_content_codes() -> None:
    records = generate_confirmatory_records(seed=0)

    for record in records:
        content_code = CONTENT_CODE_BY_ID[record.content_id]
        tokens = record.context_text.split()

        assert content_code in tokens
        assert record.content_id not in tokens

        if record.speech_act == SpeechAct.REQUEST:
            assert tokens[0] == "<REQ>"
        else:
            assert tokens[0] == "<AST>"


def test_contexts_use_valid_role_codes() -> None:
    records = generate_confirmatory_records(seed=0)
    valid_role_codes = set(ROLE_CODES.values())

    for record in records:
        tokens = record.context_text.split()

        assert tokens[1] in valid_role_codes
        assert tokens[2] in valid_role_codes


def test_lexical_transfer_contents_only_enter_lexical_ood() -> None:
    records = generate_confirmatory_records(seed=0)

    request_holdouts = set(REQUEST_LEXICAL_TRANSFER_HOLDOUTS)
    assertion_holdouts = set(ASSERTION_LEXICAL_TRANSFER_HOLDOUTS)

    for record in records:
        if (
            record.content_id in request_holdouts
            or record.content_id in assertion_holdouts
        ):
            assert record.split == Split.LEXICAL_OOD_TEST


def test_joint_request_ood_example_is_excluded() -> None:
    record = generate_experimental_record(
        content_id=REQUEST_LEXICAL_TRANSFER_HOLDOUTS[0],
        speech_act=SpeechAct.REQUEST,
        speaker_role="role_01",
        addressee_role="role_02",
        register_code=REQUEST_COMPOSITIONAL_OOD_CODES[0],
    )

    assert record is None


def test_request_targets_match_register_configuration() -> None:
    records = generate_confirmatory_records(seed=0)

    request_records = [
        record
        for record in records
        if record.speech_act == SpeechAct.REQUEST
    ]

    for record in request_records:
        register_code = record.context_text.split()[-1]
        config = COMPOSITE_REGISTER_CODES[register_code]
        content = EXPERIMENTAL_REQUEST_CONTENT[record.content_id]
        target_tokens = record.target_text.split()

        expected_verb = content.verb.select(
            config.lexical_formality
        )

        assert expected_verb in target_tokens
        assert content.object_form in target_tokens

        assert (
            INDIRECT_MARKER in target_tokens
        ) == (
            config.directness == Directness.INDIRECT
        )

        assert (
            MITIGATION_MARKER in target_tokens
        ) == (
            config.politeness_mitigation
            == PolitenessMitigation.MITIGATED
        )


def test_assertion_targets_match_register_configuration() -> None:
    records = generate_confirmatory_records(seed=0)

    assertion_records = [
        record
        for record in records
        if record.speech_act == SpeechAct.ASSERTION
    ]

    for record in assertion_records:
        register_code = record.context_text.split()[-1]
        config = COMPOSITE_REGISTER_CODES[register_code]
        content = EXPERIMENTAL_ASSERTION_CONTENT[record.content_id]
        target_tokens = record.target_text.split()

        expected_predicate = content.predicate.select(
            config.lexical_formality
        )

        assert content.subject_form in target_tokens
        assert expected_predicate in target_tokens

        assert (
            HEDGE_MARKER in target_tokens
        ) == (
            config.epistemic_stance == EpistemicStance.HEDGED
        )


def test_familiarization_has_96_records() -> None:
    records = generate_lexical_familiarization()

    assert len(records) == 96


def test_each_content_has_three_familiarization_mappings() -> None:
    records = generate_lexical_familiarization()
    counts = Counter(record.content_id for record in records)

    assert len(counts) == 32
    assert set(counts.values()) == {3}


def test_familiarization_covers_every_surface_form_once() -> None:
    records = generate_lexical_familiarization()
    targets = [record.target_text for record in records]
    expected = experimental_surface_forms()

    assert len(targets) == 96
    assert len(set(targets)) == 96
    assert set(targets) == set(expected)


def test_familiarization_contexts_are_neutral() -> None:
    records = generate_lexical_familiarization()
    register_codes = set(COMPOSITE_REGISTER_CODES)
    role_codes = set(ROLE_CODES.values())

    for record in records:
        tokens = record.context_text.split()

        assert len(tokens) == 3
        assert tokens[0] == LEXICAL_FAMILIARIZATION_CODE
        assert tokens[1] == record.content_code
        assert tokens[2] == record.variant_code.value
        assert register_codes.isdisjoint(tokens)
        assert role_codes.isdisjoint(tokens)
        assert "<REQ>" not in tokens
        assert "<AST>" not in tokens


def test_lexical_transfer_contents_are_familiarized() -> None:
    records = generate_lexical_familiarization()
    familiarized = {record.content_id for record in records}

    assert set(REQUEST_LEXICAL_TRANSFER_HOLDOUTS) <= familiarized
    assert set(ASSERTION_LEXICAL_TRANSFER_HOLDOUTS) <= familiarized


def test_familiarization_is_deterministic() -> None:
    first = generate_lexical_familiarization()
    second = generate_lexical_familiarization()

    assert first == second


def test_variant_swap_reverses_only_var0_and_var1() -> None:
    default = generate_lexical_familiarization()
    swapped = generate_lexical_familiarization(
        swap_variants=True
    )

    default_map = {
        (record.content_id, record.variant_code): record.target_text
        for record in default
    }
    swapped_map = {
        (record.content_id, record.variant_code): record.target_text
        for record in swapped
    }

    content_ids = {record.content_id for record in default}

    for content_id in content_ids:
        assert default_map[
            (content_id, FamiliarizationVariant.VAR0)
        ] == swapped_map[
            (content_id, FamiliarizationVariant.VAR1)
        ]

        assert default_map[
            (content_id, FamiliarizationVariant.VAR1)
        ] == swapped_map[
            (content_id, FamiliarizationVariant.VAR0)
        ]

        assert default_map[
            (content_id, FamiliarizationVariant.FIXED)
        ] == swapped_map[
            (content_id, FamiliarizationVariant.FIXED)
        ]
