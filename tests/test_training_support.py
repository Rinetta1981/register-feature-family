from register_feature_family.dataset_v02 import (
    generate_confirmatory_records_v02,
)
from register_feature_family.experimental_generator import (
    generate_confirmatory_records,
    generate_lexical_familiarization,
)
from register_feature_family.schemas import (
    DatasetRecord,
    Split,
)


def _record_tokens(
    context_text: str,
    target_text: str,
) -> set[str]:
    return {
        *context_text.split(),
        *target_text.split(),
    }


def _unsupported_tokens(
    records: list[DatasetRecord],
) -> dict[Split, set[str]]:
    familiarization_records = (
        generate_lexical_familiarization()
    )

    training_tokens: set[str] = set()

    for record in familiarization_records:
        training_tokens.update(
            _record_tokens(
                record.context_text,
                record.target_text,
            )
        )

    for record in records:
        if record.split == Split.TRAIN:
            training_tokens.update(
                _record_tokens(
                    record.context_text,
                    record.target_text,
                )
            )

    evaluation_splits = (
        Split.VALIDATION,
        Split.IID_TEST,
        Split.COMPOSITIONAL_OOD_TEST,
        Split.LEXICAL_OOD_TEST,
    )

    unsupported_by_split: dict[
        Split,
        set[str],
    ] = {}

    for split in evaluation_splits:
        unsupported: set[str] = set()

        for record in records:
            if record.split != split:
                continue

            unsupported.update(
                _record_tokens(
                    record.context_text,
                    record.target_text,
                )
                - training_tokens
            )

        unsupported_by_split[split] = unsupported

    return unsupported_by_split


def test_v01_preserves_known_pretraining_defect() -> None:
    records = generate_confirmatory_records(
        seed=0
    )

    assert _unsupported_tokens(records) == {
        Split.VALIDATION: set(),
        Split.IID_TEST: set(),
        Split.COMPOSITIONAL_OOD_TEST: {
            "<C03>",
            "<C04>",
        },
        Split.LEXICAL_OOD_TEST: set(),
    }


def test_v02_all_evaluation_tokens_have_training_support() -> None:
    records = generate_confirmatory_records_v02(
        seed=0
    )

    assert _unsupported_tokens(records) == {
        Split.VALIDATION: set(),
        Split.IID_TEST: set(),
        Split.COMPOSITIONAL_OOD_TEST: set(),
        Split.LEXICAL_OOD_TEST: set(),
    }
