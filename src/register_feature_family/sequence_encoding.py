from dataclasses import dataclass

from register_feature_family.experimental_generator import (
    LexicalFamiliarizationRecord,
)
from register_feature_family.schemas import DatasetRecord
from register_feature_family.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    ClosedVocabularyTokenizer,
)

IGNORE_INDEX = -100
MAX_SEQUENCE_LENGTH = 16

RESERVED_SEQUENCE_TOKENS: frozenset[str] = frozenset(
    {
        PAD_TOKEN,
        BOS_TOKEN,
        SEP_TOKEN,
        EOS_TOKEN,
    }
)


@dataclass(frozen=True)
class EncodedTrainingExample:
    """One variable-length causal-LM training example."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                "input_ids and labels must have equal length"
            )

        if not self.input_ids:
            raise ValueError(
                "encoded example must not be empty"
            )

        if len(self.input_ids) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "encoded example exceeds maximum sequence length"
            )

    @property
    def sequence_length(self) -> int:
        """Return the unpadded sequence length."""

        return len(self.input_ids)

    @property
    def supervised_token_count(self) -> int:
        """Return the number of target-side labels used for loss."""

        return sum(
            label != IGNORE_INDEX
            for label in self.labels
        )


def _validate_payload_tokens(
    *,
    tokens: tuple[str, ...],
    field_name: str,
) -> None:
    """Reject sequence-control tokens inside context or target text."""

    forbidden = RESERVED_SEQUENCE_TOKENS.intersection(tokens)

    if forbidden:
        rendered = ", ".join(sorted(forbidden))

        raise ValueError(
            f"{field_name} contains reserved sequence token: "
            f"{rendered}"
        )


def _encode_context_target(
    *,
    context_text: str,
    target_text: str,
    tokenizer: ClosedVocabularyTokenizer,
) -> EncodedTrainingExample:
    """Encode one context-target pair with target-only labels."""

    context_tokens = tuple(context_text.split())
    target_tokens = tuple(target_text.split())

    if not context_tokens:
        raise ValueError("context_text must not be empty")

    if not target_tokens:
        raise ValueError("target_text must not be empty")

    _validate_payload_tokens(
        tokens=context_tokens,
        field_name="context_text",
    )
    _validate_payload_tokens(
        tokens=target_tokens,
        field_name="target_text",
    )

    sequence_tokens = (
        BOS_TOKEN,
        *context_tokens,
        SEP_TOKEN,
        *target_tokens,
        EOS_TOKEN,
    )

    input_ids = tuple(
        tokenizer.encode_tokens(sequence_tokens)
    )

    prefix_length = (
        1
        + len(context_tokens)
        + 1
    )

    labels = (
        (IGNORE_INDEX,) * prefix_length
        + input_ids[prefix_length:]
    )

    return EncodedTrainingExample(
        input_ids=input_ids,
        labels=labels,
    )


def encode_confirmatory_record(
    record: DatasetRecord,
    *,
    tokenizer: ClosedVocabularyTokenizer | None = None,
) -> EncodedTrainingExample:
    """Encode one register-conditioned dataset record."""

    active_tokenizer = (
        tokenizer
        if tokenizer is not None
        else ClosedVocabularyTokenizer()
    )

    return _encode_context_target(
        context_text=record.context_text,
        target_text=record.target_text,
        tokenizer=active_tokenizer,
    )


def encode_familiarization_record(
    record: LexicalFamiliarizationRecord,
    *,
    tokenizer: ClosedVocabularyTokenizer | None = None,
) -> EncodedTrainingExample:
    """Encode one neutral lexical-familiarization record."""

    active_tokenizer = (
        tokenizer
        if tokenizer is not None
        else ClosedVocabularyTokenizer()
    )

    return _encode_context_target(
        context_text=record.context_text,
        target_text=record.target_text,
        tokenizer=active_tokenizer,
    )
