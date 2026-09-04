from register_feature_family.behavioral_evaluation import (
    COMPOSITIONAL_OOD_SPLIT,
    IID_SPLIT,
    LEXICAL_OOD_SPLIT,
    ComponentMetric,
    SeedBehavioralEvaluation,
    SplitBehavioralMetrics,
    aggregate_behavioral_gate,
    build_surface_to_content,
    score_prediction,
    seed_passes_behavioral_gate,
)
from register_feature_family.dataset_v02 import (
    generate_confirmatory_records_v02,
)
from register_feature_family.experimental_generator import (
    generate_lexical_familiarization,
)


def _validation_record():
    records = generate_confirmatory_records_v02(
        seed=0
    )

    return next(
        record
        for record in records
        if record.split.value == "validation"
    )


def _component(
    accuracy: float,
) -> ComponentMetric:
    denominator = 100
    correct = round(
        denominator * accuracy
    )

    return ComponentMetric(
        denominator=denominator,
        correct=correct,
        accuracy=accuracy,
    )


def _undefined_component() -> ComponentMetric:
    return ComponentMetric(
        denominator=0,
        correct=0,
        accuracy=None,
    )


def _split_metrics(
    *,
    split_name: str,
    content_accuracy: float,
    register_accuracy: float,
) -> SplitBehavioralMetrics:
    return SplitBehavioralMetrics(
        split_name=split_name,
        n_examples=100,
        content=_component(
            content_accuracy
        ),
        register_given_content=_component(
            register_accuracy
        ),
        exact_match=_component(
            register_accuracy
        ),
        lexical_formality_given_content=(
            _component(
                register_accuracy
            )
        ),
        directness_given_content=(
            _undefined_component()
        ),
        mitigation_given_content=(
            _undefined_component()
        ),
        epistemic_stance_given_content=(
            _undefined_component()
        ),
    )


def _passing_metrics() -> dict[
    str,
    SplitBehavioralMetrics,
]:
    return {
        IID_SPLIT: _split_metrics(
            split_name=IID_SPLIT,
            content_accuracy=0.99,
            register_accuracy=0.95,
        ),
        COMPOSITIONAL_OOD_SPLIT: (
            _split_metrics(
                split_name=(
                    COMPOSITIONAL_OOD_SPLIT
                ),
                content_accuracy=0.95,
                register_accuracy=0.80,
            )
        ),
        LEXICAL_OOD_SPLIT: (
            _split_metrics(
                split_name=LEXICAL_OOD_SPLIT,
                content_accuracy=0.95,
                register_accuracy=0.50,
            )
        ),
    }


def test_familiarization_recovers_three_form_lexicon() -> None:
    familiarization_records = (
        generate_lexical_familiarization(
            swap_variants=False
        )
    )

    surface_to_content = (
        build_surface_to_content(
            familiarization_records
        )
    )

    assert len(surface_to_content) == 96

    assert len(
        set(surface_to_content.values())
    ) == 32


def test_expected_validation_target_scores_perfectly() -> None:
    familiarization_records = (
        generate_lexical_familiarization(
            swap_variants=False
        )
    )

    surface_to_content = (
        build_surface_to_content(
            familiarization_records
        )
    )

    record = _validation_record()

    generated_tokens = tuple(
        record.target_text.split()
    )

    score = score_prediction(
        record=record,
        generated_tokens=generated_tokens,
        surface_to_content=surface_to_content,
    )

    assert score.content_correct
    assert score.exact_match_correct
    assert (
        score.register_correct_given_content
        is True
    )
    assert (
        score.lexical_formality_correct_given_content
        is True
    )


def test_wrong_variant_preserves_content_but_fails_register() -> None:
    familiarization_records = (
        generate_lexical_familiarization(
            swap_variants=False
        )
    )

    surface_to_content = (
        build_surface_to_content(
            familiarization_records
        )
    )

    record = _validation_record()

    expected_tokens = tuple(
        record.target_text.split()
    )

    expected_surface = next(
        token
        for token in expected_tokens
        if token in surface_to_content
    )

    content_code = surface_to_content[
        expected_surface
    ]

    alternative_surface = next(
        surface
        for surface, content in (
            surface_to_content.items()
        )
        if content == content_code
        and surface != expected_surface
    )

    generated_tokens = tuple(
        (
            alternative_surface
            if token == expected_surface
            else token
        )
        for token in expected_tokens
    )

    score = score_prediction(
        record=record,
        generated_tokens=generated_tokens,
        surface_to_content=surface_to_content,
    )

    assert score.content_correct
    assert not score.exact_match_correct
    assert (
        score.register_correct_given_content
        is False
    )
    assert (
        score.lexical_formality_correct_given_content
        is False
    )


def test_frozen_seed_gate_ignores_lexical_register_threshold() -> None:
    metrics = _passing_metrics()

    assert seed_passes_behavioral_gate(
        metrics
    )


def test_aggregate_gate_requires_two_of_three_seeds() -> None:
    passing_metrics = _passing_metrics()

    failing_metrics = _passing_metrics()

    failing_metrics[COMPOSITIONAL_OOD_SPLIT] = (
        _split_metrics(
            split_name=(
                COMPOSITIONAL_OOD_SPLIT
            ),
            content_accuracy=0.95,
            register_accuracy=0.70,
        )
    )

    seed_results = (
        SeedBehavioralEvaluation(
            seed=0,
            metrics=passing_metrics,
            passes_gate=True,
        ),
        SeedBehavioralEvaluation(
            seed=1,
            metrics=passing_metrics,
            passes_gate=True,
        ),
        SeedBehavioralEvaluation(
            seed=2,
            metrics=failing_metrics,
            passes_gate=False,
        ),
    )

    gate = aggregate_behavioral_gate(
        seed_results
    )

    assert gate.passes_gate
    assert gate.passing_seeds == (0, 1)
