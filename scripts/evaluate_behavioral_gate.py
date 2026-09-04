import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from register_feature_family.behavioral_evaluation import (
    COMPOSITIONAL_OOD_SPLIT,
    FROZEN_TRAINING_CONFIG,
    HELD_OUT_SPLIT_ORDER,
    IID_SPLIT,
    LEXICAL_OOD_SPLIT,
    PRIMARY_MODEL_SEEDS,
    PRIMARY_TOKENIZER_FINGERPRINT,
    SeedBehavioralEvaluation,
    aggregate_behavioral_gate,
    build_surface_to_content,
    evaluate_model_on_records,
    seed_passes_behavioral_gate,
)

from register_feature_family.dataset_export_v02 import (
    export_dataset_bundle_v02,
)
from register_feature_family.dataset_v02 import (
    generate_confirmatory_records_v02,
)
from register_feature_family.experimental_generator import (
    generate_lexical_familiarization,
)
from register_feature_family.tokenizer import (
    ClosedVocabularyTokenizer,
)
from register_feature_family.training import (
    PRIMARY_DATASET_FINGERPRINT,
    save_training_artifacts,
    train_development_model,
)
from register_feature_family.training_data import (
    build_training_corpus,
)

EXPECTED_HELD_OUT_COUNTS = {
    IID_SPLIT: 120,
    COMPOSITIONAL_OOD_SPLIT: 216,
    LEXICAL_OOD_SPLIT: 360,
}


def _write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the frozen three-seed behavioral gate once."""

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: evaluate_behavioral_gate.py "
            "OUTPUT_DIR"
        )

    output_dir = Path(sys.argv[1])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory() as temporary_directory:
        manifest = export_dataset_bundle_v02(
            Path(temporary_directory),
            seed=0,
            swap_variants=False,
        )

        actual_fingerprint = manifest.get(
            "dataset_fingerprint"
        )

        if (
            actual_fingerprint
            != PRIMARY_DATASET_FINGERPRINT
        ):
            raise RuntimeError(
                "dataset fingerprint mismatch: "
                f"{actual_fingerprint}"
            )

    tokenizer = ClosedVocabularyTokenizer()

    if (
        tokenizer.fingerprint
        != PRIMARY_TOKENIZER_FINGERPRINT
    ):
        raise RuntimeError(
            "tokenizer fingerprint mismatch: "
            f"{tokenizer.fingerprint}"
        )

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

    confirmatory_records = tuple(
        generate_confirmatory_records_v02(
            seed=0
        )
    )

    split_records = {
        split_name: tuple(
            record
            for record in confirmatory_records
            if record.split.value == split_name
        )
        for split_name in HELD_OUT_SPLIT_ORDER
    }

    actual_counts = {
        split_name: len(records)
        for split_name, records in (
            split_records.items()
        )
    }

    if actual_counts != EXPECTED_HELD_OUT_COUNTS:
        raise RuntimeError(
            "held-out split count mismatch: "
            f"{actual_counts}"
        )

    corpus = build_training_corpus(
        tokenizer=tokenizer
    )

    seed_results: list[
        SeedBehavioralEvaluation
    ] = []

    for seed in PRIMARY_MODEL_SEEDS:
        model, training_result = (
            train_development_model(
                corpus=corpus,
                seed=seed,
                config=FROZEN_TRAINING_CONFIG,
                tokenizer=tokenizer,
            )
        )

        seed_dir = (
            output_dir
            / f"seed-{seed}"
        )

        save_training_artifacts(
            output_dir=seed_dir,
            model=model,
            result=training_result,
        )

        metrics_by_split = {}
        prediction_rows: list[
            dict[str, object]
        ] = []

        for split_name in HELD_OUT_SPLIT_ORDER:
            metrics, scores = (
                evaluate_model_on_records(
                    model=model,
                    records=(
                        split_records[
                            split_name
                        ]
                    ),
                    tokenizer=tokenizer,
                    surface_to_content=(
                        surface_to_content
                    ),
                    split_name=split_name,
                )
            )

            metrics_by_split[
                split_name
            ] = metrics

            for score in scores:
                prediction_row = {
                    "split": split_name,
                    **score.to_json_dict(),
                }

                prediction_rows.append(
                    prediction_row
                )

        passes_gate = (
            seed_passes_behavioral_gate(
                metrics_by_split
            )
        )

        seed_result = (
            SeedBehavioralEvaluation(
                seed=seed,
                metrics=metrics_by_split,
                passes_gate=passes_gate,
            )
        )

        seed_results.append(
            seed_result
        )

        _write_json(
            seed_dir
            / "behavioral_metrics.json",
            seed_result.to_json_dict(),
        )

        predictions_path = (
            seed_dir
            / "predictions.jsonl"
        )

        predictions_path.write_text(
            "\n".join(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                for row in prediction_rows
            )
            + "\n",
            encoding="utf-8",
        )

    gate = aggregate_behavioral_gate(
        seed_results
    )

    final_payload: dict[str, object] = {
        "dataset_fingerprint": (
            PRIMARY_DATASET_FINGERPRINT
        ),
        "tokenizer_fingerprint": (
            PRIMARY_TOKENIZER_FINGERPRINT
        ),
        "gate": gate.to_json_dict(),
        "seeds": [
            result.to_json_dict()
            for result in seed_results
        ],
    }

    _write_json(
        output_dir
        / "behavioral_gate.json",
        final_payload,
    )

    print(
        json.dumps(
            final_payload,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
