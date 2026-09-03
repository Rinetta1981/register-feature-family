from register_feature_family.experimental_generator import (
    generate_confirmatory_records,
    generate_lexical_familiarization,
)
from register_feature_family.schemas import Split


def _record_tokens(
    context_text: str,
    target_text: str,
) -> set[str]:
    return {
        *context_text.split(),
        *target_text.split(),
    }


def test_all_evaluation_tokens_have_training_support() -> None:
    confirmatory_records = generate_confirmatory_records(
        seed=0
    )
    familiarization_records = generate_lexical_familiarization()

    training_tokens: set[str] = set()

    for record in familiarization_records:
        training_tokens.update(
            _record_tokens(
                record.context_text,
                record.target_text,
            )
        )

    for record in confirmatory_records:
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

        for record in confirmatory_records:
            if record.split != split:
                continue

            record_tokens = _record_tokens(
                record.context_text,
                record.target_text,
            )

            unsupported.update(
                record_tokens - training_tokens
            )

        unsupported_by_split[split] = unsupported

    assert unsupported_by_split == {
        Split.VALIDATION: set(),
        Split.IID_TEST: set(),
        Split.COMPOSITIONAL_OOD_TEST: set(),
        Split.LEXICAL_OOD_TEST: set(),
    }
