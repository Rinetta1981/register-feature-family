import json
from pathlib import Path

import torch

from register_feature_family.tokenizer import (
    ClosedVocabularyTokenizer,
)
from register_feature_family.training import (
    PRIMARY_DATASET_FINGERPRINT,
    TrainingConfig,
    save_training_artifacts,
    train_development_model,
)
from register_feature_family.training_data import (
    TrainingCorpus,
    build_training_corpus,
    collate_training_examples,
)


def test_training_corpus_uses_only_development_material() -> None:
    corpus = build_training_corpus()

    assert len(
        corpus.familiarization_examples
    ) == 96

    assert len(
        corpus.register_train_examples
    ) == 840

    assert len(
        corpus.validation_examples
    ) == 120

    assert len(
        corpus.validation_records
    ) == 120


def test_collation_pads_to_model_length() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    corpus = build_training_corpus(
        tokenizer=tokenizer
    )

    examples = (
        corpus.familiarization_examples[0],
        corpus.register_train_examples[0],
    )

    batch = collate_training_examples(
        examples,
        tokenizer=tokenizer,
    )

    assert batch.input_ids.shape == (2, 16)
    assert batch.labels.shape == (2, 16)
    assert batch.attention_mask.shape == (2, 16)

    first_length = examples[0].sequence_length

    assert int(
        batch.attention_mask[0].sum().item()
    ) == first_length

    assert torch.all(
        batch.input_ids[
            0,
            first_length:,
        ]
        == tokenizer.pad_token_id
    )


def test_small_training_run_is_finite(
    tmp_path: Path,
) -> None:
    corpus = build_training_corpus()

    small_corpus = TrainingCorpus(
        familiarization_examples=(
            corpus.familiarization_examples[:8]
        ),
        register_train_examples=(
            corpus.register_train_examples[:16]
        ),
        validation_examples=(
            corpus.validation_examples[:4]
        ),
        validation_records=(
            corpus.validation_records[:4]
        ),
    )

    config = TrainingConfig(
        batch_size=8,
        familiarization_epochs=1,
        register_epochs=1,
        minimum_register_epochs=1,
        early_stopping_patience=1,
    )

    model, result = train_development_model(
        corpus=small_corpus,
        seed=0,
        config=config,
    )

    assert result.dataset_fingerprint == (
        PRIMARY_DATASET_FINGERPRINT
    )

    assert result.parameter_count == 438272

    assert torch.isfinite(
        torch.tensor(
            result.familiarization_final_loss
        )
    )

    assert torch.isfinite(
        torch.tensor(
            result.best_validation_loss
        )
    )

    assert (
        0.0
        <= result.validation_exact_match
        <= 1.0
    )

    save_training_artifacts(
        output_dir=tmp_path,
        model=model,
        result=result,
    )

    assert (
        tmp_path / "model.pt"
    ).is_file()

    metrics_path = tmp_path / "metrics.json"

    assert metrics_path.is_file()

    metrics = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )

    assert metrics[
        "dataset_fingerprint"
    ] == PRIMARY_DATASET_FINGERPRINT
