import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from register_feature_family.model import (
    DecoderOnlyTransformer,
    causal_lm_loss,
)
from register_feature_family.schemas import DatasetRecord
from register_feature_family.sequence_encoding import (
    IGNORE_INDEX,
    EncodedTrainingExample,
)
from register_feature_family.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN,
    ClosedVocabularyTokenizer,
)
from register_feature_family.training_data import (
    TrainingCorpus,
    iter_training_batches,
)

PRIMARY_DATASET_FINGERPRINT = (
    "ff1315604a48d991a97af8a1b8ba8749f5f9bff94614263a155d216c7770e8c7"
)


@dataclass(frozen=True)
class TrainingConfig:
    """Validation-only development hyperparameters."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    familiarization_epochs: int = 150
    register_epochs: int = 80
    minimum_register_epochs: int = 20
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-5

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive"
            )

        if self.familiarization_epochs < 1:
            raise ValueError(
                "familiarization_epochs must be positive"
            )

        if self.register_epochs < 1:
            raise ValueError(
                "register_epochs must be positive"
            )

        if self.minimum_register_epochs < 1:
            raise ValueError(
                "minimum_register_epochs must be positive"
            )

        if (
            self.minimum_register_epochs
            > self.register_epochs
        ):
            raise ValueError(
                "minimum_register_epochs exceeds register_epochs"
            )

        if self.early_stopping_patience < 1:
            raise ValueError(
                "early_stopping_patience must be positive"
            )


@dataclass(frozen=True)
class TrainingRunResult:
    """Summary of one validation-only development run."""

    model_seed: int
    familiarization_final_loss: float
    best_validation_loss: float
    best_validation_epoch: int
    register_epochs_completed: int
    validation_exact_match: float
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    parameter_count: int
    source_revision: str
    training_config: TrainingConfig

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable run summary."""

        return {
            "model_seed": self.model_seed,
            "familiarization_final_loss": (
                self.familiarization_final_loss
            ),
            "best_validation_loss": (
                self.best_validation_loss
            ),
            "best_validation_epoch": (
                self.best_validation_epoch
            ),
            "register_epochs_completed": (
                self.register_epochs_completed
            ),
            "validation_exact_match": (
                self.validation_exact_match
            ),
            "dataset_fingerprint": (
                self.dataset_fingerprint
            ),
            "tokenizer_fingerprint": (
                self.tokenizer_fingerprint
            ),
            "parameter_count": self.parameter_count,
            "source_revision": self.source_revision,
            "training_config": {
                "batch_size": (
                    self.training_config.batch_size
                ),
                "learning_rate": (
                    self.training_config.learning_rate
                ),
                "weight_decay": (
                    self.training_config.weight_decay
                ),
                "gradient_clip_norm": (
                    self.training_config.gradient_clip_norm
                ),
                "familiarization_epochs": (
                    self.training_config.familiarization_epochs
                ),
                "register_epochs": (
                    self.training_config.register_epochs
                ),
                "minimum_register_epochs": (
                    self.training_config.minimum_register_epochs
                ),
                "early_stopping_patience": (
                    self.training_config.early_stopping_patience
                ),
                "early_stopping_min_delta": (
                    self.training_config.early_stopping_min_delta
                ),
            },
        }


def _snapshot_state(
    model: DecoderOnlyTransformer,
) -> dict[str, torch.Tensor]:
    """Copy model parameters for validation checkpoint selection."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _supervised_token_count(
    labels: torch.Tensor,
) -> int:
    """Count labels that contribute to shifted causal loss."""

    shifted_labels = labels[:, 1:]

    return int(
        torch.count_nonzero(
            shifted_labels != IGNORE_INDEX
        ).item()
    )


def _train_epoch(
    *,
    model: DecoderOnlyTransformer,
    examples: tuple[EncodedTrainingExample, ...],
    optimizer: torch.optim.Optimizer,
    tokenizer: ClosedVocabularyTokenizer,
    config: TrainingConfig,
    shuffle_seed: int,
) -> float:
    """Train for one deterministic epoch."""

    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch in iter_training_batches(
        examples,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
        shuffle=True,
        seed=shuffle_seed,
    ):
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
        )

        loss = causal_lm_loss(
            logits=logits,
            labels=batch.labels,
        )

        supervised_tokens = _supervised_token_count(
            batch.labels
        )

        torch.autograd.backward(loss)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
        )

        optimizer.step()

        total_loss += (
            float(loss.item())
            * supervised_tokens
        )
        total_tokens += supervised_tokens

    if total_tokens == 0:
        raise RuntimeError(
            "training epoch contained no supervised tokens"
        )

    return total_loss / total_tokens


def _evaluate_loss(
    *,
    model: DecoderOnlyTransformer,
    examples: tuple[EncodedTrainingExample, ...],
    tokenizer: ClosedVocabularyTokenizer,
    batch_size: int,
) -> float:
    """Evaluate target-only loss without parameter updates."""

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in iter_training_batches(
            examples,
            batch_size=batch_size,
            tokenizer=tokenizer,
            shuffle=False,
            seed=0,
        ):
            logits = model(
                batch.input_ids,
                attention_mask=batch.attention_mask,
            )

            loss = causal_lm_loss(
                logits=logits,
                labels=batch.labels,
            )

            supervised_tokens = (
                _supervised_token_count(
                    batch.labels
                )
            )

            total_loss += (
                float(loss.item())
                * supervised_tokens
            )
            total_tokens += supervised_tokens

    if total_tokens == 0:
        raise RuntimeError(
            "evaluation contained no supervised tokens"
        )

    return total_loss / total_tokens


def greedy_generate_target(
    *,
    model: DecoderOnlyTransformer,
    record: DatasetRecord,
    tokenizer: ClosedVocabularyTokenizer,
) -> tuple[str, ...]:
    """Greedily generate one target using context only."""

    prefix_tokens = (
        BOS_TOKEN,
        *record.context_text.split(),
        SEP_TOKEN,
    )

    token_ids = tokenizer.encode_tokens(
        prefix_tokens
    )

    generated_tokens: list[str] = []

    model.eval()

    with torch.no_grad():
        while (
            len(token_ids)
            < model.config.max_sequence_length
        ):
            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
            )

            logits = model(input_ids)

            next_token_id = int(
                torch.argmax(
                    logits[0, -1, :]
                ).item()
            )

            next_token = tokenizer.token_for_id(
                next_token_id
            )

            if next_token == EOS_TOKEN:
                break

            generated_tokens.append(
                next_token
            )

            token_ids.append(
                next_token_id
            )

    return tuple(generated_tokens)


def evaluate_exact_match(
    *,
    model: DecoderOnlyTransformer,
    records: tuple[DatasetRecord, ...],
    tokenizer: ClosedVocabularyTokenizer,
) -> float:
    """Calculate greedy exact-target validation accuracy."""

    if not records:
        raise ValueError(
            "validation records must not be empty"
        )

    correct = 0

    for record in records:
        generated = greedy_generate_target(
            model=model,
            record=record,
            tokenizer=tokenizer,
        )

        expected = tuple(
            record.target_text.split()
        )

        if generated == expected:
            correct += 1

    return correct / len(records)


def train_development_model(
    *,
    corpus: TrainingCorpus,
    seed: int,
    config: TrainingConfig | None = None,
    tokenizer: ClosedVocabularyTokenizer | None = None,
) -> tuple[
    DecoderOnlyTransformer,
    TrainingRunResult,
]:
    """Train one model using only training and validation data."""

    active_config = (
        config
        if config is not None
        else TrainingConfig()
    )

    active_tokenizer = (
        tokenizer
        if tokenizer is not None
        else ClosedVocabularyTokenizer()
    )

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = DecoderOnlyTransformer()

    familiarization_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )

    familiarization_final_loss = math.inf

    for epoch in range(
        1,
        active_config.familiarization_epochs + 1,
    ):
        familiarization_final_loss = _train_epoch(
            model=model,
            examples=corpus.familiarization_examples,
            optimizer=familiarization_optimizer,
            tokenizer=active_tokenizer,
            config=active_config,
            shuffle_seed=seed * 100_000 + epoch,
        )

    register_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )

    joint_training_examples = (
        corpus.register_train_examples
        + corpus.familiarization_examples
    )

    best_validation_loss = math.inf
    best_validation_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(
        1,
        active_config.register_epochs + 1,
    ):
        completed_epochs = epoch

        _train_epoch(
            model=model,
            examples=joint_training_examples,
            optimizer=register_optimizer,
            tokenizer=active_tokenizer,
            config=active_config,
            shuffle_seed=(
                seed * 1_000_000
                + 500_000
                + epoch
            ),
        )

        validation_loss = _evaluate_loss(
            model=model,
            examples=corpus.validation_examples,
            tokenizer=active_tokenizer,
            batch_size=active_config.batch_size,
        )

        if (
            validation_loss
            < best_validation_loss
            - active_config.early_stopping_min_delta
        ):
            best_validation_loss = validation_loss
            best_validation_epoch = epoch
            best_state = _snapshot_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch
            >= active_config.minimum_register_epochs
            and epochs_without_improvement
            >= active_config.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "training produced no validation checkpoint"
        )

    model.load_state_dict(best_state)

    validation_exact_match = evaluate_exact_match(
        model=model,
        records=corpus.validation_records,
        tokenizer=active_tokenizer,
    )

    result = TrainingRunResult(
        model_seed=seed,
        familiarization_final_loss=(
            familiarization_final_loss
        ),
        best_validation_loss=(
            best_validation_loss
        ),
        best_validation_epoch=(
            best_validation_epoch
        ),
        register_epochs_completed=(
            completed_epochs
        ),
        validation_exact_match=(
            validation_exact_match
        ),
        dataset_fingerprint=(
            PRIMARY_DATASET_FINGERPRINT
        ),
        tokenizer_fingerprint=(
            active_tokenizer.fingerprint
        ),
        parameter_count=(
            model.trainable_parameter_count()
        ),
        source_revision=os.environ.get(
            "GITHUB_SHA",
            "local",
        ),
        training_config=active_config,
    )

    return model, result


def save_training_artifacts(
    *,
    output_dir: Path,
    model: DecoderOnlyTransformer,
    result: TrainingRunResult,
) -> None:
    """Save checkpoint and human-readable training metrics."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = result.to_json_dict()

    metrics_path = output_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            metrics,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": model.config.vocab_size,
            "max_sequence_length": (
                model.config.max_sequence_length
            ),
            "n_layers": model.config.n_layers,
            "n_heads": model.config.n_heads,
            "d_model": model.config.d_model,
            "d_mlp": model.config.d_mlp,
        },
        "run_metadata": metrics,
    }

    torch.save(
        checkpoint,
        output_dir / "model.pt",
    )
