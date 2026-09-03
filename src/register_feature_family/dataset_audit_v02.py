import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
)

from register_feature_family.dataset_export_v02 import (
    V01_FAMILIARIZATION_SHA256,
)
from register_feature_family.experimental_generator import (
    LexicalFamiliarizationRecord,
)
from register_feature_family.request_controls import (
    REQUEST_COMPOSITE_CODES,
    REQUEST_CONTROL_BY_COMPOSITE_CODE,
    REQUEST_CONTROL_TOKENS,
    request_control_tokens,
)
from register_feature_family.schemas import (
    DatasetRecord,
    SpeechAct,
    Split,
)

ModelT = TypeVar(
    "ModelT",
    bound=BaseModel,
)

EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    Split.TRAIN.value: 840,
    Split.VALIDATION.value: 120,
    Split.IID_TEST.value: 120,
    Split.COMPOSITIONAL_OOD_TEST.value: 216,
    Split.LEXICAL_OOD_TEST.value: 360,
}


class ManifestFileV02(BaseModel):
    """One file entry in the v0.2 manifest."""

    model_config = ConfigDict(extra="forbid")

    records: int
    sha256: str


class DatasetManifestV02(BaseModel):
    """Typed representation of the v0.2 manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    dataset_version: str
    dataset_seed: int
    swap_variants: bool
    confirmatory_records: int
    familiarization_records: int
    split_counts: dict[str, int]
    files: dict[str, ManifestFileV02]
    dataset_fingerprint: str
    supersedes_dataset_version: str
    supersedes_fingerprint: str


@dataclass(frozen=True)
class DatasetAuditV02Report:
    """Result of independently auditing one v0.2 bundle."""

    passed: bool
    issues: tuple[str, ...]


def _read_jsonl(
    path: Path,
    model_type: type[ModelT],
) -> list[ModelT]:
    """Read and validate one JSONL file."""

    lines = [
        line
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]

    return [
        model_type.model_validate_json(line)
        for line in lines
    ]


def _sha256_file(path: Path) -> str:
    """Calculate a file hash independently."""

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _canonical_json(value: object) -> str:
    """Create deterministic JSON for fingerprint verification."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _audit_fingerprint(
    *,
    confirmatory_hash: str,
    familiarization_hash: str,
    dataset_seed: int,
    swap_variants: bool,
) -> str:
    """Recalculate the v0.2 fingerprint independently."""

    payload = {
        "dataset_version": "0.2",
        "dataset_seed": dataset_seed,
        "swap_variants": swap_variants,
        "confirmatory_sha256": confirmatory_hash,
        "lexical_familiarization_sha256": familiarization_hash,
    }

    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _record_tokens(
    context_text: str,
    target_text: str,
) -> set[str]:
    """Return all model-facing tokens in one record."""

    return {
        *context_text.split(),
        *target_text.split(),
    }


def audit_dataset_bundle_v02(
    output_dir: Path,
) -> DatasetAuditV02Report:
    """Independently audit a frozen dataset v0.2 bundle."""

    issues: list[str] = []

    confirmatory_path = (
        output_dir / "confirmatory.jsonl"
    )
    familiarization_path = (
        output_dir / "lexical_familiarization.jsonl"
    )
    manifest_path = output_dir / "manifest.json"

    required_paths = (
        confirmatory_path,
        familiarization_path,
        manifest_path,
    )

    missing = [
        path.name
        for path in required_paths
        if not path.is_file()
    ]

    if missing:
        return DatasetAuditV02Report(
            passed=False,
            issues=(
                f"missing files: {sorted(missing)}",
            ),
        )

    try:
        manifest = (
            DatasetManifestV02.model_validate_json(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        )
        confirmatory_records = _read_jsonl(
            confirmatory_path,
            DatasetRecord,
        )
        familiarization_records = _read_jsonl(
            familiarization_path,
            LexicalFamiliarizationRecord,
        )
    except (OSError, ValidationError) as error:
        return DatasetAuditV02Report(
            passed=False,
            issues=(str(error),),
        )

    if manifest.schema_version != "0.2":
        issues.append(
            "manifest schema version is not 0.2"
        )

    if manifest.dataset_version != "0.2":
        issues.append(
            "manifest dataset version is not 0.2"
        )

    if len(confirmatory_records) != 1656:
        issues.append(
            "confirmatory record count is not 1656"
        )

    if len(familiarization_records) != 96:
        issues.append(
            "familiarization record count is not 96"
        )

    actual_split_counts = {
        split.value: sum(
            confirmatory_record.split == split
            for confirmatory_record in confirmatory_records
        )
        for split in Split
    }

    if actual_split_counts != EXPECTED_SPLIT_COUNTS:
        issues.append(
            "confirmatory split counts are incorrect"
        )

    if manifest.split_counts != EXPECTED_SPLIT_COUNTS:
        issues.append(
            "manifest split counts are incorrect"
        )

    confirmatory_hash = _sha256_file(
        confirmatory_path
    )
    familiarization_hash = _sha256_file(
        familiarization_path
    )

    confirmatory_file = manifest.files.get(
        "confirmatory.jsonl"
    )
    familiarization_file = manifest.files.get(
        "lexical_familiarization.jsonl"
    )

    if confirmatory_file is None:
        issues.append(
            "manifest lacks confirmatory.jsonl"
        )
    else:
        if confirmatory_file.records != 1656:
            issues.append(
                "manifest confirmatory count is incorrect"
            )

        if confirmatory_file.sha256 != confirmatory_hash:
            issues.append(
                "confirmatory hash mismatch"
            )

    if familiarization_file is None:
        issues.append(
            "manifest lacks lexical_familiarization.jsonl"
        )
    else:
        if familiarization_file.records != 96:
            issues.append(
                "manifest familiarization count is incorrect"
            )

        if familiarization_file.sha256 != familiarization_hash:
            issues.append(
                "familiarization hash mismatch"
            )

    if (
        not manifest.swap_variants
        and familiarization_hash
        != V01_FAMILIARIZATION_SHA256
    ):
        issues.append(
            "unchanged familiarization file differs from v0.1"
        )

    expected_fingerprint = _audit_fingerprint(
        confirmatory_hash=confirmatory_hash,
        familiarization_hash=familiarization_hash,
        dataset_seed=manifest.dataset_seed,
        swap_variants=manifest.swap_variants,
    )

    if (
        manifest.dataset_fingerprint
        != expected_fingerprint
    ):
        issues.append(
            "dataset fingerprint mismatch"
        )

    known_control_combinations = set(
        REQUEST_CONTROL_BY_COMPOSITE_CODE.values()
    )

    training_control_combinations: set[
        tuple[str, str, str]
    ] = set()

    training_control_tokens: set[str] = set()
    opaque_request_codes_seen: set[str] = set()

    invalid_control_combinations: set[
        tuple[str, str, str]
    ] = set()

    for confirmatory_record in confirmatory_records:
        if (
            confirmatory_record.speech_act
            != SpeechAct.REQUEST
        ):
            continue

        context_tokens = tuple(
            confirmatory_record.context_text.split()
        )

        opaque_request_codes_seen.update(
            set(context_tokens).intersection(
                REQUEST_COMPOSITE_CODES
            )
        )

        if len(context_tokens) < 3:
            issues.append(
                "request context is too short for v0.2 controls"
            )
            continue

        controls = (
            context_tokens[-3],
            context_tokens[-2],
            context_tokens[-1],
        )

        if controls not in known_control_combinations:
            invalid_control_combinations.add(
                controls
            )

        if confirmatory_record.split == Split.TRAIN:
            training_control_combinations.add(
                controls
            )
            training_control_tokens.update(
                controls
            )

    if opaque_request_codes_seen:
        issues.append(
            "opaque request codes remain model-facing: "
            f"{sorted(opaque_request_codes_seen)}"
        )

    if invalid_control_combinations:
        issues.append(
            "invalid request control combinations found"
        )

    if training_control_tokens != set(
        REQUEST_CONTROL_TOKENS
    ):
        issues.append(
            "not all request control tokens have training support"
        )

    held_combinations = {
        request_control_tokens("<C03>"),
        request_control_tokens("<C04>"),
    }

    if (
        training_control_combinations
        & held_combinations
    ):
        issues.append(
            "compositional-OOD control combination leaked into training"
        )

    training_tokens: set[str] = set()

    for familiarization_record in familiarization_records:
        training_tokens.update(
            _record_tokens(
                familiarization_record.context_text,
                familiarization_record.target_text,
            )
        )

    for confirmatory_record in confirmatory_records:
        if confirmatory_record.split == Split.TRAIN:
            training_tokens.update(
                _record_tokens(
                    confirmatory_record.context_text,
                    confirmatory_record.target_text,
                )
            )

    evaluation_splits = (
        Split.VALIDATION,
        Split.IID_TEST,
        Split.COMPOSITIONAL_OOD_TEST,
        Split.LEXICAL_OOD_TEST,
    )

    unsupported_by_split: dict[
        Split,
        set[str],
    ] = {
        split: set()
        for split in evaluation_splits
    }

    for confirmatory_record in confirmatory_records:
        if (
            confirmatory_record.split
            not in evaluation_splits
        ):
            continue

        unsupported_by_split[
            confirmatory_record.split
        ].update(
            _record_tokens(
                confirmatory_record.context_text,
                confirmatory_record.target_text,
            )
            - training_tokens
        )

    for split, unsupported in (
        unsupported_by_split.items()
    ):
        if unsupported:
            issues.append(
                f"unsupported tokens in {split.value}: "
                f"{sorted(unsupported)}"
            )

    return DatasetAuditV02Report(
        passed=not issues,
        issues=tuple(issues),
    )
