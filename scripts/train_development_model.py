import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from register_feature_family.dataset_export_v02 import (
    export_dataset_bundle_v02,
)
from register_feature_family.training import (
    PRIMARY_DATASET_FINGERPRINT,
    save_training_artifacts,
    train_development_model,
)
from register_feature_family.training_data import (
    build_training_corpus,
)


def main() -> None:
    """Train one validation-only development model."""

    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: train_development_model.py "
            "MODEL_SEED OUTPUT_DIR"
        )

    seed = int(sys.argv[1])
    output_dir = Path(sys.argv[2])

    with TemporaryDirectory() as temporary_directory:
        manifest = export_dataset_bundle_v02(
            Path(temporary_directory),
            seed=0,
            swap_variants=False,
        )

        actual_fingerprint = manifest.get(
            "dataset_fingerprint"
        )

        if actual_fingerprint != PRIMARY_DATASET_FINGERPRINT:
            raise RuntimeError(
                "dataset fingerprint mismatch: "
                f"{actual_fingerprint}"
            )

    corpus = build_training_corpus()

    model, result = train_development_model(
        corpus=corpus,
        seed=seed,
    )

    save_training_artifacts(
        output_dir=output_dir,
        model=model,
        result=result,
    )

    print(
        json.dumps(
            result.to_json_dict(),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
