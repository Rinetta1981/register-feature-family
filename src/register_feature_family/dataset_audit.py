import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from register_feature_family.codebook import (
    COMPOSITE_REGISTER_CODES,
    ROLE_CODES,
)
from register_feature_family.content_inventory import CONTENT_CODE_BY_ID
from register_feature_family.dataset_export import (
    CONFIRMATORY_FILENAME,
    FAMILIARIZATION_FILENAME,
    MANIFEST_FILENAME,
)
from register_feature_family.experimental_codebook import (
    experimental_surface_forms,
)
from register_feature_family.experimental_generator import (
    LEXICAL_FAMILIARIZATION_CODE,
    FamiliarizationVariant,
    LexicalFamiliarizationRecord,
)
from register_feature_family.schemas import (
    DatasetRecord,
    SpeechAct,
    Split,
)
from register_feature_family.split_config import (
    ASSERTION_LEXICAL_TRANSFER_HOLDOUTS,
    REQUEST_COMPOSITIONAL_OOD_FEATURES,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
)

EXPECTED_CONFIRMATORY_RECORDS = 1656
EXPECTED_FAMILIARIZATION_RECORDS = 96

EXPECTED_SPLIT_COUNTS: dict[Split, int] = {
    Split.TRAIN: 840,
    Split.VALIDATION: 120,
    Split.IID_TEST: 120,
    Split.COMPOSITIONAL_OOD_TEST: 216,
    Split.LEXICAL_OOD_TEST: 360,
}


class ManifestFile(BaseModel):
    """Metadata for one exported dataset file."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    records: int


class DatasetManifest(BaseModel):
    """Schema for the deterministic export manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"]
    dataset_seed: int
    swap_variants: bool
    confirmatory_records: int
    familiarization_records: int
    split_counts: dict[str, int]
    files: dict[str, ManifestFile]
    dataset_fingerprint: str


class DatasetAuditReport(BaseModel):
    """Result of auditing one exported dataset bundle."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: tuple[str, ...]
    confirmatory_records: int
    familiarization_records: int
    split_counts: dict[str, int]
    confirmatory_sha256: str
    familiarization_sha256: str
    dataset_fingerprint: str


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _bundle_fingerprint(
    *,
    confirmatory_sha256: str,
    familiarization_sha256: str,
    seed: int,
    swap_variants: bool,
) -> str:
    """Recompute the complete dataset fingerprint independently."""

    value = (
        f"confirmatory={confirmatory_sha256}\n"
        f"familiarization={familiarization_sha256}\n"
        f"seed={seed}\n"
        f"swap_variants={swap_variants}\n"
    )

    return hashlib.sha256(value.encode()).hexdigest()


def _read_confirmatory(path: Path) -> list[DatasetRecord]:
    """Read and validate confirmatory JSONL records."""

    return [
        DatasetRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_familiarization(
    path: Path,
) -> list[LexicalFamiliarizationRecord]:
    """Read and validate lexical-familiarization JSONL records."""

    return [
        LexicalFamiliarizationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _check(
    condition: bool,
    message: str,
    issues: list[str],
) -> None:
    """Record an audit issue when one condition fails."""

    if not condition:
        issues.append(message)


def audit_dataset_bundle(
    *,
    output_dir: Path,
) -> DatasetAuditReport:
    """Audit one exported dataset bundle against the frozen protocol."""

    confirmatory_path = output_dir / CONFIRMATORY_FILENAME
    familiarization_path = output_dir / FAMILIARIZATION_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    confirmatory = _read_confirmatory(confirmatory_path)
    familiarization = _read_familiarization(familiarization_path)

    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    issues: list[str] = []

    split_counter = Counter(
        record.split
        for record in confirmatory
    )

    split_counts = {
        split.value: split_counter[split]
        for split in Split
    }

    _check(
        len(confirmatory) == EXPECTED_CONFIRMATORY_RECORDS,
        "confirmatory record count does not equal 1656",
        issues,
    )

    _check(
        len(familiarization) == EXPECTED_FAMILIARIZATION_RECORDS,
        "familiarization record count does not equal 96",
        issues,
    )

    _check(
        split_counter == Counter(EXPECTED_SPLIT_COUNTS),
        "confirmatory split counts do not match the frozen protocol",
        issues,
    )

    confirmatory_ids = [
        record.example_id
        for record in confirmatory
    ]

    familiarization_ids = [
        record.example_id
        for record in familiarization
    ]

    _check(
        len(confirmatory_ids) == len(set(confirmatory_ids)),
        "confirmatory example IDs are not globally unique",
        issues,
    )

    _check(
        len(familiarization_ids) == len(set(familiarization_ids)),
        "familiarization example IDs are not globally unique",
        issues,
    )

    lexical_holdouts = (
        set(REQUEST_LEXICAL_TRANSFER_HOLDOUTS)
        | set(ASSERTION_LEXICAL_TRANSFER_HOLDOUTS)
    )

    lexical_leaks = [
        record.example_id
        for record in confirmatory
        if record.content_id in lexical_holdouts
        and record.split != Split.LEXICAL_OOD_TEST
    ]

    _check(
        not lexical_leaks,
        "lexical-transfer content leaked into a non-lexical split",
        issues,
    )

    ordinary_splits = {
        Split.TRAIN,
        Split.VALIDATION,
        Split.IID_TEST,
    }

    group_splits: dict[str, set[Split]] = defaultdict(set)

    for record in confirmatory:
        if record.split in ordinary_splits:
            group_splits[record.comparison_group_id].add(
                record.split
            )

    split_groups = [
        group_id
        for group_id, splits in group_splits.items()
        if len(splits) != 1
    ]

    _check(
        not split_groups,
        "an ordinary comparison group crosses split boundaries",
        issues,
    )

    frozen_request_features = set(
        REQUEST_COMPOSITIONAL_OOD_FEATURES
    )

    compositional_errors: list[str] = []

    for record in confirmatory:
        if record.speech_act == SpeechAct.ASSERTION:
            if record.split == Split.COMPOSITIONAL_OOD_TEST:
                compositional_errors.append(record.example_id)
            continue

        feature_bundle = (
            record.lexical_formality,
            record.directness,
            record.politeness_mitigation,
        )

        is_frozen_bundle = (
            feature_bundle in frozen_request_features
        )

        is_lexical_holdout = (
            record.content_id
            in REQUEST_LEXICAL_TRANSFER_HOLDOUTS
        )

        invalid_compositional_membership = (
            (
                is_frozen_bundle
                and is_lexical_holdout
            )
            or (
                is_frozen_bundle
                and record.split
                != Split.COMPOSITIONAL_OOD_TEST
            )
            or (
                not is_frozen_bundle
                and record.split
                == Split.COMPOSITIONAL_OOD_TEST
            )
        )

        if invalid_compositional_membership:
            compositional_errors.append(record.example_id)

    _check(
        not compositional_errors,
        "compositional-OOD membership violates frozen rules",
        issues,
    )

    familiarization_targets = [
        record.target_text
        for record in familiarization
    ]

    _check(
        len(set(familiarization_targets))
        == EXPECTED_FAMILIARIZATION_RECORDS,
        "familiarization target forms are not globally unique",
        issues,
    )

    _check(
        set(familiarization_targets)
        == set(experimental_surface_forms()),
        "familiarization does not cover all 96 surface forms",
        issues,
    )

    variants_by_content: dict[
        str,
        set[FamiliarizationVariant],
    ] = defaultdict(set)

    neutral_errors: list[str] = []

    register_codes = set(COMPOSITE_REGISTER_CODES)
    role_codes = set(ROLE_CODES.values())

    for record in familiarization:
        variants_by_content[record.content_id].add(
            record.variant_code
        )

        tokens = record.context_text.split()

        expected_content_code = CONTENT_CODE_BY_ID.get(
            record.content_id
        )

        if (
            len(tokens) != 3
            or tokens[0] != LEXICAL_FAMILIARIZATION_CODE
            or record.content_code != expected_content_code
            or tokens[1] != record.content_code
            or tokens[2] != record.variant_code.value
            or not register_codes.isdisjoint(tokens)
            or not role_codes.isdisjoint(tokens)
            or "<REQ>" in tokens
            or "<AST>" in tokens
        ):
            neutral_errors.append(record.example_id)

    expected_variants = set(FamiliarizationVariant)

    _check(
        len(variants_by_content) == 32
        and all(
            variants == expected_variants
            for variants in variants_by_content.values()
        ),
        "each content must have VAR0, VAR1, and FIXED mappings",
        issues,
    )

    _check(
        not neutral_errors,
        "familiarization contains non-neutral control information",
        issues,
    )

    confirmatory_sha256 = _sha256_file(confirmatory_path)
    familiarization_sha256 = _sha256_file(
        familiarization_path
    )

    fingerprint = _bundle_fingerprint(
        confirmatory_sha256=confirmatory_sha256,
        familiarization_sha256=familiarization_sha256,
        seed=manifest.dataset_seed,
        swap_variants=manifest.swap_variants,
    )

    confirmatory_manifest = manifest.files.get(
        CONFIRMATORY_FILENAME
    )
    familiarization_manifest = manifest.files.get(
        FAMILIARIZATION_FILENAME
    )

    _check(
        confirmatory_manifest is not None,
        "manifest is missing confirmatory file metadata",
        issues,
    )

    _check(
        familiarization_manifest is not None,
        "manifest is missing familiarization file metadata",
        issues,
    )

    if confirmatory_manifest is not None:
        _check(
            confirmatory_manifest.sha256
            == confirmatory_sha256,
            "confirmatory SHA-256 does not match manifest",
            issues,
        )
        _check(
            confirmatory_manifest.records
            == len(confirmatory),
            "confirmatory manifest count is incorrect",
            issues,
        )

    if familiarization_manifest is not None:
        _check(
            familiarization_manifest.sha256
            == familiarization_sha256,
            "familiarization SHA-256 does not match manifest",
            issues,
        )
        _check(
            familiarization_manifest.records
            == len(familiarization),
            "familiarization manifest count is incorrect",
            issues,
        )

    _check(
        manifest.confirmatory_records == len(confirmatory),
        "top-level confirmatory manifest count is incorrect",
        issues,
    )

    _check(
        manifest.familiarization_records
        == len(familiarization),
        "top-level familiarization manifest count is incorrect",
        issues,
    )

    _check(
        manifest.split_counts == split_counts,
        "manifest split counts do not match exported data",
        issues,
    )

    _check(
        manifest.dataset_fingerprint == fingerprint,
        "dataset fingerprint does not match exported files",
        issues,
    )

    return DatasetAuditReport(
        passed=not issues,
        issues=tuple(issues),
        confirmatory_records=len(confirmatory),
        familiarization_records=len(familiarization),
        split_counts=split_counts,
        confirmatory_sha256=confirmatory_sha256,
        familiarization_sha256=familiarization_sha256,
        dataset_fingerprint=fingerprint,
    )
