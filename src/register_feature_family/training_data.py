from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from random import Random

import torch

from register_feature_family.dataset_v02 import (
    generate_confirmatory_records_v02,
)
from register_feature_family.experimental_generator import (
    generate_lexical_familiarization,
)
from register_feature_family.schemas import DatasetRecord, Split
from register_feature_family.sequence_encoding import (
    IGNORE_INDEX,
    MAX_SEQUENCE_LENGTH,
    EncodedTrainingExample,
    encode_confirmatory_record,
    encode_familiarization_record,
)
from register_feature_family.tokenizer import ClosedVocabularyTokenizer


@dataclass(frozen=True)
class TrainingBatch:
    """One padded model-training batch."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class TrainingCorpus:
    """Training and validation material available during development."""

    familiarization_examples: tuple[EncodedTrainingExample, ...]
    register_train_examples: tuple[EncodedTrainingExample, ...]
    validation_examples: tuple[EncodedTrainingExample, ...]
    validation_records: tuple[DatasetRecord, ...]


def collate_training_examples(
    examples: Sequence[EncodedTrainingExample],
    *,
    tokenizer: ClosedVocabularyTokenizer,
) -> TrainingBatch:
    """Pad encoded examples to the fixed model sequence length."""

    if not examples:
        raise ValueError("cannot collate an empty batch")

    batch_size = len(examples)

    input_ids = torch.full(
        (batch_size, MAX_SEQUENCE_LENGTH),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, MAX_SEQUENCE_LENGTH),
        IGNORE_INDEX,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, MAX_SEQUENCE_LENGTH),
        dtype=torch.long,
    )

    for row, example in enumerate(examples):
        length = example.sequence_length

        input_ids[row, :length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
        )

        labels[row, :length] = torch.tensor(
            example.labels,
            dtype=torch.long,
        )

        attention_mask[row, :length] = 1

    return TrainingBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


def iter_training_batches(
    examples: Sequence[EncodedTrainingExample],
    *,
    batch_size: int,
    tokenizer: ClosedVocabularyTokenizer,
    shuffle: bool,
    seed: int,
) -> Iterator[TrainingBatch]:
    """Yield deterministic padded batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    indices = list(range(len(examples)))

    if shuffle:
        Random(seed).shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[
            start : start + batch_size
        ]

        batch_examples = [
            examples[index]
            for index in batch_indices
        ]

        yield collate_training_examples(
            batch_examples,
            tokenizer=tokenizer,
        )


def build_training_corpus(
    *,
    tokenizer: ClosedVocabularyTokenizer | None = None,
) -> TrainingCorpus:
    """Build only the training and validation development corpus."""

    active_tokenizer = (
        tokenizer
        if tokenizer is not None
        else ClosedVocabularyTokenizer()
    )

    confirmatory_records = (
        generate_confirmatory_records_v02(seed=0)
    )

    register_train_records = tuple(
        record
        for record in confirmatory_records
        if record.split == Split.TRAIN
    )

    validation_records = tuple(
        record
        for record in confirmatory_records
        if record.split == Split.VALIDATION
    )

    familiarization_records = (
        generate_lexical_familiarization(
            swap_variants=False
        )
    )

    familiarization_examples = tuple(
        encode_familiarization_record(
            record,
            tokenizer=active_tokenizer,
        )
        for record in familiarization_records
    )

    register_train_examples = tuple(
        encode_confirmatory_record(
            record,
            tokenizer=active_tokenizer,
        )
        for record in register_train_records
    )

    validation_examples = tuple(
        encode_confirmatory_record(
            record,
            tokenizer=active_tokenizer,
        )
        for record in validation_records
    )

    return TrainingCorpus(
        familiarization_examples=familiarization_examples,
        register_train_examples=register_train_examples,
        validation_examples=validation_examples,
        validation_records=validation_records,
    )
