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
    LEXICAL_FAMILIARIZATION_CODE,
    FamiliarizationVariant,
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


def test_confirmatory_dataset_matches_protocol() -> None:
    records = generate_confirmatory_records(seed=0)

    assert len(records) == 1656

    counts = Counter(record.split for record in records)

    assert counts == {
        Split.TRAIN: 840,
        Split.VALIDATION: 120,
        Split.IID_TEST: 120,
        Split.COMPOSITIONAL_OOD_TEST: 216,
        Split.LEXICAL_OOD_TEST: 360,
    }

    example_ids = [record.example_id for record in records]

    assert len(example_ids) == len(set(example_ids))


def test_confirmatory_contexts_use_neutral_codes() -> None:
    records = generate_confirmatory_records(seed=0)
    valid_role_codes = set(ROLE_CODES.values())

    for record in records:
        tokens = record.context_text.split()
        content_code = CONTENT_CODE_BY_ID[record.content_id]

        assert content_code in tokens
        assert record.content_id not in tokens
        assert tokens[1] in valid_role_codes
        assert tokens[2] in valid_role_codes

        if record.speech_act == SpeechAct.REQUEST:
            assert tokens[0] == "<REQ>"
        else:
            assert tokens[0] == "<AST>"


def test_lexical_transfer_items_do_not_leak() -> None:
    records = generate_confirmatory_records(seed=0)

    holdouts = (
        set(REQUEST_LEXICAL_TRANSFER_HOLDOUTS)
        | set(ASSERTION_LEXICAL_TRANSFER_HOLDOUTS)
    )

    for record in records:
        if record.content_id in holdouts:
            assert record.split == Split.LEXICAL_OOD_TEST


def test_joint_request_ood_is_excluded() -> None:
    record = generate_experimental_record(
        content_id=REQUEST_LEXICAL_TRANSFER_HOLDOUTS[0],
        speech_act=SpeechAct.REQUEST,
        speaker_role="role_01",
        addressee_role="role_02",
        register_code=REQUEST_COMPOSITIONAL_OOD_CODES[0],
    )

    assert record is None


def test_request_targets_match_register_features() -> None:
    records = generate_confirmatory_records(seed=0)

    for record in records:
        if record.speech_act != SpeechAct.REQUEST:
            continue

        register_code = record.context_text.split()[-1]
        config = COMPOSITE_REGISTER_CODES[register_code]
        content = EXPERIMENTAL_REQUEST_CONTENT[record.content_id]
        tokens = record.target_text.split()

        assert content.verb.select(config.lexical_formality) in tokens
        assert content.object_form in tokens

        assert (
            INDIRECT_MARKER in tokens
        ) == (
            config.directness == Directness.INDIRECT
        )

        assert (
            MITIGATION_MARKER in tokens
        ) == (
            config.politeness_mitigation
            == PolitenessMitigation.MITIGATED
        )


def test_assertion_targets_match_register_features() -> None:
    records = generate_confirmatory_records(seed=0)

    for record in records:
        if record.speech_act != SpeechAct.ASSERTION:
            continue

        register_code = record.context_text.split()[-1]
        config = COMPOSITE_REGISTER_CODES[register_code]
        content = EXPERIMENTAL_ASSERTION_CONTENT[record.content_id]
        tokens = record.target_text.split()

        assert content.subject_form in tokens
        assert content.predicate.select(
            config.lexical_formality
        ) in tokens

        assert (
            HEDGE_MARKER in tokens
        ) == (
            config.epistemic_stance == EpistemicStance.HEDGED
        )


def test_familiarization_has_complete_neutral_coverage() -> None:
    records = generate_lexical_familiarization()

    assert len(records) == 96

    content_counts = Counter(
        record.content_id for record in records
    )

    assert len(content_counts) == 32
    assert set(content_counts.values()) == {3}

    targets = [record.target_text for record in records]

    assert len(set(targets)) == 96
    assert set(targets) == set(experimental_surface_forms())

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


def test_variant_swap_reverses_var0_and_var1_only() -> None:
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
