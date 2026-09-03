from pathlib import Path

from register_feature_family.dataset_audit_v02 import (
    DatasetManifestV02,
    audit_dataset_bundle_v02,
)
from register_feature_family.dataset_export_v02 import (
    V01_CONFIRMATORY_SHA256,
    V01_DATASET_FINGERPRINT,
    V01_FAMILIARIZATION_SHA256,
    export_dataset_bundle_v02,
)


def test_exported_v02_bundle_passes_audit(
    tmp_path: Path,
) -> None:
    export_dataset_bundle_v02(
        tmp_path
    )

    report = audit_dataset_bundle_v02(
        tmp_path
    )

    assert report.passed
    assert report.issues == ()


def test_v02_export_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    export_dataset_bundle_v02(first)
    export_dataset_bundle_v02(second)

    filenames = (
        "confirmatory.jsonl",
        "lexical_familiarization.jsonl",
        "manifest.json",
    )

    for filename in filenames:
        assert (
            first.joinpath(filename).read_bytes()
            == second.joinpath(filename).read_bytes()
        )


def test_v02_has_new_dataset_identity(
    tmp_path: Path,
) -> None:
    manifest_data = export_dataset_bundle_v02(
        tmp_path
    )

    manifest = DatasetManifestV02.model_validate(
        manifest_data
    )

    assert (
        manifest.dataset_fingerprint
        != V01_DATASET_FINGERPRINT
    )

    assert (
        manifest.files[
            "confirmatory.jsonl"
        ].sha256
        != V01_CONFIRMATORY_SHA256
    )


def test_v02_preserves_familiarization_bytes(
    tmp_path: Path,
) -> None:
    manifest_data = export_dataset_bundle_v02(
        tmp_path
    )

    manifest = DatasetManifestV02.model_validate(
        manifest_data
    )

    assert (
        manifest.files[
            "lexical_familiarization.jsonl"
        ].sha256
        == V01_FAMILIARIZATION_SHA256
    )
