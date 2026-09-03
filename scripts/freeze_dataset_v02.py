import json
from pathlib import Path

from register_feature_family.dataset_audit_v02 import (
    audit_dataset_bundle_v02,
)
from register_feature_family.dataset_export_v02 import (
    export_dataset_bundle_v02,
)

OUTPUT_DIR = Path("data/frozen/v0.2")
DATASET_SEED = 0


def main() -> None:
    """Build, audit, and report frozen dataset v0.2."""

    manifest = export_dataset_bundle_v02(
        OUTPUT_DIR,
        seed=DATASET_SEED,
        swap_variants=False,
    )

    report = audit_dataset_bundle_v02(
        OUTPUT_DIR
    )

    if not report.passed:
        print("Dataset v0.2 audit failed:")

        for issue in report.issues:
            print(f"- {issue}")

        raise SystemExit(1)

    print(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
