from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from register_feature_family.model import (
    DecoderOnlyTransformer,
)
from register_feature_family.schemas import DatasetRecord
from register_feature_family.tokenizer import (
    ClosedVocabularyTokenizer,
)
from register_feature_family.training import (
    TrainingConfig,
    greedy_generate_target,
)

IID_SPLIT = "iid_test"
COMPOSITIONAL_OOD_SPLIT = "compositional_ood_test"
LEXICAL_OOD_SPLIT = "lexical_ood_test"

HELD_OUT_SPLIT_ORDER = (
    IID_SPLIT,
    COMPOSITIONAL_OOD_SPLIT,
    LEXICAL_OOD_SPLIT,
)

PRIMARY_MODEL_SEEDS = (0, 1, 2)

PRIMARY_TOKENIZER_FINGERPRINT = (
    "f1cf46bd23a6b4870bfdfc4192cff98b7d7fc20706052c2a1aee9c278b793eef"
)

IID_CONTENT_THRESHOLD = 0.95
OOD_CONTENT_THRESHOLD = 0.90
IID_REGISTER_THRESHOLD = 0.90
COMPOSITIONAL_REGISTER_THRESHOLD = 0.75
MINIMUM_PASSING_SEEDS = 2

REQUEST_TOKEN = "<REQ>"
ASSERTION_TOKEN = "<AST>"

INDIRECT_MARKER = "kelo"
MITIGATION_MARKER = "mira"
EPISTEMIC_HEDGE_MARKER = "sava"

FROZEN_TRAINING_CONFIG = TrainingConfig(
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    familiarization_epochs=150,
    register_epochs=80,
    minimum_register_epochs=20,
    early_stopping_patience=12,
    early_stopping_min_delta=1e-5,
)


@dataclass(frozen=True)
class ComponentMetric:
    """Correct count, denominator, and derived accuracy."""

    denominator: int
    correct: int
    accuracy: float | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metric."""

        return {
            "denominator": self.denominator,
            "correct": self.correct,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class PredictionScore:
    """Behavioral scoring for one generated target."""

    context_text: str
    expected_target: str
    generated_target: str
    content_correct: bool
    exact_match_correct: bool
    register_correct_given_content: bool | None
    lexical_formality_correct_given_content: bool | None
    directness_correct_given_content: bool | None
    mitigation_correct_given_content: bool | None
    epistemic_stance_correct_given_content: bool | None

    def to_json_dict(self) -> dict[str, object]:
        """Return an auditable prediction record."""

        return {
            "context_text": self.context_text,
            "expected_target": self.expected_target,
            "generated_target": self.generated_target,
            "content_correct": self.content_correct,
            "exact_match_correct": self.exact_match_correct,
            "register_correct_given_content": (
                self.register_correct_given_content
            ),
            "lexical_formality_correct_given_content": (
                self.lexical_formality_correct_given_content
            ),
            "directness_correct_given_content": (
                self.directness_correct_given_content
            ),
            "mitigation_correct_given_content": (
                self.mitigation_correct_given_content
            ),
            "epistemic_stance_correct_given_content": (
                self.epistemic_stance_correct_given_content
            ),
        }


@dataclass(frozen=True)
class SplitBehavioralMetrics:
    """Behavioral metrics for one confirmatory split."""

    split_name: str
    n_examples: int
    content: ComponentMetric
    register_given_content: ComponentMetric
    exact_match: ComponentMetric
    lexical_formality_given_content: ComponentMetric
    directness_given_content: ComponentMetric
    mitigation_given_content: ComponentMetric
    epistemic_stance_given_content: ComponentMetric

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-serializable split metrics."""

        return {
            "split_name": self.split_name,
            "n_examples": self.n_examples,
            "content": self.content.to_json_dict(),
            "register_given_content": (
                self.register_given_content.to_json_dict()
            ),
            "exact_match": (
                self.exact_match.to_json_dict()
            ),
            "lexical_formality_given_content": (
                self.lexical_formality_given_content.to_json_dict()
            ),
            "directness_given_content": (
                self.directness_given_content.to_json_dict()
            ),
            "mitigation_given_content": (
                self.mitigation_given_content.to_json_dict()
            ),
            "epistemic_stance_given_content": (
                self.epistemic_stance_given_content.to_json_dict()
            ),
        }


@dataclass(frozen=True)
class SeedBehavioralEvaluation:
    """Confirmatory behavioral result for one model seed."""

    seed: int
    metrics: dict[str, SplitBehavioralMetrics]
    passes_gate: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-serializable seed results."""

        return {
            "seed": self.seed,
            "passes_gate": self.passes_gate,
            "metrics": {
                split_name: metric.to_json_dict()
                for split_name, metric in self.metrics.items()
            },
        }


@dataclass(frozen=True)
class BehavioralGateSummary:
    """Aggregate preregistered 2-of-3 seed gate."""

    passing_seeds: tuple[int, ...]
    required_passing_seeds: int
    passes_gate: bool

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-serializable aggregate gate."""

        return {
            "passing_seeds": list(self.passing_seeds),
            "passing_seed_count": len(
                self.passing_seeds
            ),
            "required_passing_seeds": (
                self.required_passing_seeds
            ),
            "passes_gate": self.passes_gate,
            "thresholds": {
                "iid_content": IID_CONTENT_THRESHOLD,
                "ood_content": OOD_CONTENT_THRESHOLD,
                "iid_register_given_content": (
                    IID_REGISTER_THRESHOLD
                ),
                "compositional_register_given_content": (
                    COMPOSITIONAL_REGISTER_THRESHOLD
                ),
            },
        }


def _content_code_from_context(
    context_text: str,
) -> str:
    """Extract the single model-facing content code."""

    matches = tuple(
        token
        for token in context_text.split()
        if token.startswith("<CONTENT_")
        and token.endswith(">")
    )

    if len(matches) != 1:
        raise ValueError(
            "expected exactly one content code in context"
        )

    return matches[0]


def build_surface_to_content(
    familiarization_records: Sequence[DatasetRecord],
) -> dict[str, str]:
    """Recover the frozen three-form semantic lexicon."""

    surface_to_content: dict[str, str] = {}

    for record in familiarization_records:
        content_code = _content_code_from_context(
            record.context_text
        )

        target_tokens = tuple(
            record.target_text.split()
        )

        if len(target_tokens) != 1:
            raise ValueError(
                "familiarization target must contain "
                "exactly one surface form"
            )

        surface_form = target_tokens[0]

        if surface_form in surface_to_content:
            raise ValueError(
                "duplicate familiarization surface form"
            )

        surface_to_content[
            surface_form
        ] = content_code

    if len(surface_to_content) != 96:
        raise ValueError(
            "expected 96 familiarized surface forms"
        )

    content_counts: dict[str, int] = {}

    for content_code in surface_to_content.values():
        content_counts[content_code] = (
            content_counts.get(content_code, 0)
            + 1
        )

    if len(content_counts) != 32:
        raise ValueError(
            "expected 32 semantic content families"
        )

    if any(
        count != 3
        for count in content_counts.values()
    ):
        raise ValueError(
            "each content family must have three forms"
        )

    return surface_to_content


def _surface_forms(
    tokens: Sequence[str],
    surface_to_content: Mapping[str, str],
) -> tuple[str, ...]:
    """Return all familiarized lexical forms in token order."""

    return tuple(
        token
        for token in tokens
        if token in surface_to_content
    )


def score_prediction(
    *,
    record: DatasetRecord,
    generated_tokens: tuple[str, ...],
    surface_to_content: Mapping[str, str],
) -> PredictionScore:
    """Score one prediction without using split identity."""

    expected_tokens = tuple(
        record.target_text.split()
    )

    expected_content = _content_code_from_context(
        record.context_text
    )

    expected_surfaces = _surface_forms(
        expected_tokens,
        surface_to_content,
    )

    if not expected_surfaces:
        raise ValueError(
            "expected target must contain at least "
            "one known surface form"
        )

    if any(
        surface_to_content[surface]
        != expected_content
        for surface in expected_surfaces
    ):
        raise ValueError(
            "expected target contains a surface form "
            "from the wrong content family"
        )

    generated_surfaces = _surface_forms(
        generated_tokens,
        surface_to_content,
    )

    content_correct = (
        len(generated_surfaces)
        == len(expected_surfaces)
        and all(
            surface_to_content[surface]
            == expected_content
            for surface in generated_surfaces
        )
    )

    exact_match_correct = (
        generated_tokens == expected_tokens
    )

    context_tokens = set(
        record.context_text.split()
    )

    is_request = REQUEST_TOKEN in context_tokens
    is_assertion = ASSERTION_TOKEN in context_tokens

    if is_request == is_assertion:
        raise ValueError(
            "context must identify exactly one speech act"
        )

    if content_correct:
        register_correct = exact_match_correct

        lexical_formality_correct = (
            generated_surfaces
            == expected_surfaces
        )

        if is_request:
            directness_correct = (
                (
                    INDIRECT_MARKER
                    in generated_tokens
                )
                == (
                    INDIRECT_MARKER
                    in expected_tokens
                )
            )

            mitigation_correct = (
                (
                    MITIGATION_MARKER
                    in generated_tokens
                )
                == (
                    MITIGATION_MARKER
                    in expected_tokens
                )
            )

            epistemic_stance_correct = None
        else:
            directness_correct = None
            mitigation_correct = None

            epistemic_stance_correct = (
                (
                    EPISTEMIC_HEDGE_MARKER
                    in generated_tokens
                )
                == (
                    EPISTEMIC_HEDGE_MARKER
                    in expected_tokens
                )
            )
    else:
        register_correct = None
        lexical_formality_correct = None
        directness_correct = None
        mitigation_correct = None
        epistemic_stance_correct = None

    return PredictionScore(
        context_text=record.context_text,
        expected_target=record.target_text,
        generated_target=" ".join(
            generated_tokens
        ),
        content_correct=content_correct,
        exact_match_correct=exact_match_correct,
        register_correct_given_content=(
            register_correct
        ),
        lexical_formality_correct_given_content=(
            lexical_formality_correct
        ),
        directness_correct_given_content=(
            directness_correct
        ),
        mitigation_correct_given_content=(
            mitigation_correct
        ),
        epistemic_stance_correct_given_content=(
            epistemic_stance_correct
        ),
    )


def _defined_values(
    values: Sequence[bool | None],
) -> tuple[bool, ...]:
    """Remove undefined conditional component scores."""

    return tuple(
        value
        for value in values
        if value is not None
    )


def _component_metric(
    values: Sequence[bool],
) -> ComponentMetric:
    """Summarize Boolean outcomes."""

    denominator = len(values)
    correct = sum(
        1
        for value in values
        if value
    )

    accuracy = (
        correct / denominator
        if denominator > 0
        else None
    )

    return ComponentMetric(
        denominator=denominator,
        correct=correct,
        accuracy=accuracy,
    )


def summarize_prediction_scores(
    *,
    split_name: str,
    scores: Sequence[PredictionScore],
) -> SplitBehavioralMetrics:
    """Aggregate prediction scores for one split."""

    if not scores:
        raise ValueError(
            "cannot summarize an empty split"
        )

    content_values = tuple(
        score.content_correct
        for score in scores
    )

    exact_match_values = tuple(
        score.exact_match_correct
        for score in scores
    )

    register_values = _defined_values(
        tuple(
            score.register_correct_given_content
            for score in scores
        )
    )

    lexical_values = _defined_values(
        tuple(
            score.lexical_formality_correct_given_content
            for score in scores
        )
    )

    directness_values = _defined_values(
        tuple(
            score.directness_correct_given_content
            for score in scores
        )
    )

    mitigation_values = _defined_values(
        tuple(
            score.mitigation_correct_given_content
            for score in scores
        )
    )

    stance_values = _defined_values(
        tuple(
            score.epistemic_stance_correct_given_content
            for score in scores
        )
    )

    return SplitBehavioralMetrics(
        split_name=split_name,
        n_examples=len(scores),
        content=_component_metric(
            content_values
        ),
        register_given_content=(
            _component_metric(
                register_values
            )
        ),
        exact_match=_component_metric(
            exact_match_values
        ),
        lexical_formality_given_content=(
            _component_metric(
                lexical_values
            )
        ),
        directness_given_content=(
            _component_metric(
                directness_values
            )
        ),
        mitigation_given_content=(
            _component_metric(
                mitigation_values
            )
        ),
        epistemic_stance_given_content=(
            _component_metric(
                stance_values
            )
        ),
    )


def evaluate_model_on_records(
    *,
    model: DecoderOnlyTransformer,
    records: Sequence[DatasetRecord],
    tokenizer: ClosedVocabularyTokenizer,
    surface_to_content: Mapping[str, str],
    split_name: str,
) -> tuple[
    SplitBehavioralMetrics,
    tuple[PredictionScore, ...],
]:
    """Generate and score a complete evaluation split."""

    scores = tuple(
        score_prediction(
            record=record,
            generated_tokens=(
                greedy_generate_target(
                    model=model,
                    record=record,
                    tokenizer=tokenizer,
                )
            ),
            surface_to_content=surface_to_content,
        )
        for record in records
    )

    metrics = summarize_prediction_scores(
        split_name=split_name,
        scores=scores,
    )

    return metrics, scores


def _required_accuracy(
    metric: ComponentMetric,
    *,
    metric_name: str,
) -> float:
    """Require an accuracy for a gated component."""

    if metric.accuracy is None:
        raise ValueError(
            f"{metric_name} has no denominator"
        )

    return metric.accuracy


def seed_passes_behavioral_gate(
    metrics: Mapping[
        str,
        SplitBehavioralMetrics,
    ],
) -> bool:
    """Apply the frozen seed-level behavioral gate."""

    try:
        iid = metrics[IID_SPLIT]
        compositional = metrics[
            COMPOSITIONAL_OOD_SPLIT
        ]
        lexical = metrics[
            LEXICAL_OOD_SPLIT
        ]
    except KeyError as error:
        raise ValueError(
            "missing required behavioral split"
        ) from error

    iid_content = _required_accuracy(
        iid.content,
        metric_name="IID content",
    )

    compositional_content = _required_accuracy(
        compositional.content,
        metric_name="compositional OOD content",
    )

    lexical_content = _required_accuracy(
        lexical.content,
        metric_name="lexical OOD content",
    )

    iid_register = _required_accuracy(
        iid.register_given_content,
        metric_name="IID register",
    )

    compositional_register = _required_accuracy(
        compositional.register_given_content,
        metric_name="compositional OOD register",
    )

    return (
        iid_content >= IID_CONTENT_THRESHOLD
        and compositional_content
        >= OOD_CONTENT_THRESHOLD
        and lexical_content
        >= OOD_CONTENT_THRESHOLD
        and iid_register
        >= IID_REGISTER_THRESHOLD
        and compositional_register
        >= COMPOSITIONAL_REGISTER_THRESHOLD
    )


def aggregate_behavioral_gate(
    results: Sequence[SeedBehavioralEvaluation],
) -> BehavioralGateSummary:
    """Apply the frozen 2-of-3 primary-seed rule."""

    if len(results) != len(
        PRIMARY_MODEL_SEEDS
    ):
        raise ValueError(
            "expected exactly three primary seed results"
        )

    result_seeds = tuple(
        result.seed
        for result in results
    )

    if (
        len(set(result_seeds))
        != len(PRIMARY_MODEL_SEEDS)
        or set(result_seeds)
        != set(PRIMARY_MODEL_SEEDS)
    ):
        raise ValueError(
            "primary seed results must be seeds 0, 1, and 2"
        )

    for result in results:
        expected_gate = (
            seed_passes_behavioral_gate(
                result.metrics
            )
        )

        if result.passes_gate != expected_gate:
            raise ValueError(
                "seed gate flag does not match metrics"
            )

    passing_seeds = tuple(
        sorted(
            result.seed
            for result in results
            if result.passes_gate
        )
    )

    passes_gate = (
        len(passing_seeds)
        >= MINIMUM_PASSING_SEEDS
    )

    return BehavioralGateSummary(
        passing_seeds=passing_seeds,
        required_passing_seeds=(
            MINIMUM_PASSING_SEEDS
        ),
        passes_gate=passes_gate,
    )
