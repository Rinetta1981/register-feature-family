import pytest

from register_feature_family.experimental_generator import (
    generate_confirmatory_records,
    generate_lexical_familiarization,
)
from register_feature_family.sequence_encoding import (
    IGNORE_INDEX,
    MAX_SEQUENCE_LENGTH,
    EncodedTrainingExample,
    encode_confirmatory_record,
    encode_familiarization_record,
)
from register_feature_family.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN,
    ClosedVocabularyTokenizer,
)


def test_confirmatory_sequence_matches_protocol() -> None:
    tokenizer = ClosedVocabularyTokenizer()
    record = generate_confirmatory_records(seed=0)[0]

    encoded = encode_confirmatory_record(
        record,
        tokenizer=tokenizer,
    )

    decoded = tokenizer.decode_ids(
        encoded.input_ids
    )

    expected = (
        BOS_TOKEN,
        *record.context_text.split(),
        SEP_TOKEN,
        *record.target_text.split(),
        EOS_TOKEN,
    )

    assert decoded == expected


def test_familiarization_sequence_matches_protocol() -> None:
    tokenizer = ClosedVocabularyTokenizer()
    record = generate_lexical_familiarization()[0]

    encoded = encode_familiarization_record(
        record,
        tokenizer=tokenizer,
    )

    decoded = tokenizer.decode_ids(
        encoded.input_ids
    )

    expected = (
        BOS_TOKEN,
        *record.context_text.split(),
        SEP_TOKEN,
        *record.target_text.split(),
        EOS_TOKEN,
    )

    assert decoded == expected


def test_confirmatory_loss_is_target_only() -> None:
    tokenizer = ClosedVocabularyTokenizer()
    record = generate_confirmatory_records(seed=0)[0]

    encoded = encode_confirmatory_record(
        record,
        tokenizer=tokenizer,
    )

    context_length = len(
        record.context_text.split()
    )

    prefix_length = (
        1
        + context_length
        + 1
    )

    assert encoded.labels[:prefix_length] == (
        IGNORE_INDEX,
    ) * prefix_length

    expected_target_ids = tuple(
        tokenizer.encode_tokens(
            (
                *record.target_text.split(),
                EOS_TOKEN,
            )
        )
    )

    assert (
        encoded.labels[prefix_length:]
        == expected_target_ids
    )


def test_first_target_token_is_supervised() -> None:
    tokenizer = ClosedVocabularyTokenizer()
    record = generate_confirmatory_records(seed=0)[0]

    encoded = encode_confirmatory_record(
        record,
        tokenizer=tokenizer,
    )

    sep_id = tokenizer.sep_token_id
    sep_position = encoded.input_ids.index(sep_id)

    first_target_position = sep_position + 1

    assert (
        encoded.labels[first_target_position]
        == encoded.input_ids[first_target_position]
    )


def test_supervised_count_is_target_plus_eos() -> None:
    record = generate_confirmatory_records(seed=0)[0]

    encoded = encode_confirmatory_record(record)

    expected = (
        len(record.target_text.split())
        + 1
    )

    assert encoded.supervised_token_count == expected


def test_all_confirmatory_sequences_fit_length_limit() -> None:
    records = generate_confirmatory_records(seed=0)

    lengths = [
        encode_confirmatory_record(
            record
        ).sequence_length
        for record in records
    ]

    assert max(lengths) <= MAX_SEQUENCE_LENGTH
    assert max(lengths) == 12


def test_all_familiarization_sequences_fit_length_limit() -> None:
    records = generate_lexical_familiarization()

    lengths = [
        encode_familiarization_record(
            record
        ).sequence_length
        for record in records
    ]

    assert max(lengths) <= MAX_SEQUENCE_LENGTH
    assert set(lengths) == {7}


def test_encoding_is_deterministic() -> None:
    record = generate_confirmatory_records(seed=0)[0]

    first = encode_confirmatory_record(record)
    second = encode_confirmatory_record(record)

    assert first == second


def test_reserved_sequence_token_in_context_is_rejected() -> None:
    record = generate_lexical_familiarization()[0]

    corrupted = record.model_copy(
        update={
            "context_text": (
                f"{record.context_text} <EOS>"
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="reserved sequence token",
    ):
        encode_familiarization_record(corrupted)


def test_encoded_example_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        EncodedTrainingExample(
            input_ids=(1, 2, 3),
            labels=(1, 2),
        )


def test_encoded_example_rejects_overlong_sequence() -> None:
    overlong = tuple(
        range(MAX_SEQUENCE_LENGTH + 1)
    )

    with pytest.raises(
        ValueError,
        match="exceeds maximum sequence length",
    ):
        EncodedTrainingExample(
            input_ids=overlong,
            labels=overlong,
        )
