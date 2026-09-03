import json
from pathlib import Path

from register_feature_family.dataset_audit import audit_dataset_bundle
from register_feature_family.dataset_export import (
    CONFIRMATORY_FILENAME,
    FAMILIARIZATION_FILENAME,
    MANIFEST_FILENAME,
    export_dataset_bundle,
)
from register_feature_family.schemas import Split


def test_exported_bundle_passes_audit(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"

    manifest = export_dataset_bundle(
        output_dir=output_dir,
        seed=0,
    )

    report = audit_dataset_bundle(
        output_dir=output_dir,
    )

    assert report.passed
    assert report.issues == ()
    assert report.confirmatory_records == 1656
    assert report.familiarization_records == 96

    assert report.split_counts == {
        Split.TRAIN.value: 840,
        Split.VALIDATION.value: 120,
        Split.IID_TEST.value: 120,
        Split.COMPOSITIONAL_OOD_TEST.value: 216,
        Split.LEXICAL_OOD_TEST.value: 360,
    }

    assert (
        report.dataset_fingerprint
        == manifest["dataset_fingerprint"]
    )


def test_export_is_byte_for_byte_deterministic(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_manifest = export_dataset_bundle(
        output_dir=first_dir,
        seed=0,
    )
    second_manifest = export_dataset_bundle(
        output_dir=second_dir,
        seed=0,
    )

    assert first_manifest == second_manifest

    filenames = (
        CONFIRMATORY_FILENAME,
        FAMILIARIZATION_FILENAME,
        MANIFEST_FILENAME,
    )

    for filename in filenames:
        assert (
            first_dir / filename
        ).read_bytes() == (
            second_dir / filename
        ).read_bytes()


def test_audit_detects_confirmatory_file_tampering(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "dataset"

    export_dataset_bundle(
        output_dir=output_dir,
        seed=0,
    )

    confirmatory_path = (
        output_dir / CONFIRMATORY_FILENAME
    )

    lines = confirmatory_path.read_text(
        encoding="utf-8"
    ).splitlines()

    first_record = json.loads(lines[0])

    first_record["target_text"] = (
        f"{first_record['target_text']} tampered"
    )

    lines[0] = json.dumps(
        first_record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    tampered_text = "\n".join(lines) + "\n"

    confirmatory_path.write_text(
        tampered_text,
        encoding="utf-8",
        newline="\n",
    )

    report = audit_dataset_bundle(
        output_dir=output_dir,
    )

    assert not report.passed

    assert (
        "confirmatory SHA-256 does not match manifest"
        in report.issues
    )

    assert (
        "dataset fingerprint does not match exported files"
        in report.issues
    )


def test_variant_swap_changes_only_familiarization_artifact(
    tmp_path: Path,
) -> None:
    default_dir = tmp_path / "default"
    swapped_dir = tmp_path / "swapped"

    default_manifest = export_dataset_bundle(
        output_dir=default_dir,
        seed=0,
        swap_variants=False,
    )

    swapped_manifest = export_dataset_bundle(
        output_dir=swapped_dir,
        seed=0,
        swap_variants=True,
    )

    default_report = audit_dataset_bundle(
        output_dir=default_dir,
    )
    swapped_report = audit_dataset_bundle(
        output_dir=swapped_dir,
    )

    assert default_report.passed
    assert swapped_report.passed

    assert (
        default_manifest["files"][CONFIRMATORY_FILENAME]["sha256"]
        == swapped_manifest["files"][CONFIRMATORY_FILENAME]["sha256"]
    )

    assert (
        default_manifest["files"][FAMILIARIZATION_FILENAME]["sha256"]
        != swapped_manifest["files"][FAMILIARIZATION_FILENAME]["sha256"]
    )

    assert (
        default_manifest["dataset_fingerprint"]
        != swapped_manifest["dataset_fingerprint"]
    )
