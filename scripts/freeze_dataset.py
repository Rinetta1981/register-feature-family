import json
from pathlib import Path

from register_feature_family.dataset_audit import audit_dataset_bundle
from register_feature_family.dataset_export import export_dataset_bundle

OUTPUT_DIR = Path("data/frozen/v0.1")
DATASET_SEED = 0


def main() -> None:
    """Export, audit, and report the frozen confirmatory dataset."""

    export_dataset_bundle(
        output_dir=OUTPUT_DIR,
        seed=DATASET_SEED,
        swap_variants=False,
    )

    report = audit_dataset_bundle(
        output_dir=OUTPUT_DIR,
    )

    if not report.passed:
        print("Dataset audit failed:")

        for issue in report.issues:
            print(f"- {issue}")

        raise SystemExit(1)

    summary = {
        "audit_passed": report.passed,
        "confirmatory_records": report.confirmatory_records,
        "familiarization_records": report.familiarization_records,
        "split_counts": report.split_counts,
        "confirmatory_sha256": report.confirmatory_sha256,
        "familiarization_sha256": report.familiarization_sha256,
        "dataset_fingerprint": report.dataset_fingerprint,
    }

    print("Frozen dataset created successfully.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
