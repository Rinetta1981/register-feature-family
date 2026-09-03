import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from register_feature_family.dataset_v02 import (
    DATASET_VERSION_V02,
    generate_confirmatory_records_v02,
)
from register_feature_family.experimental_generator import (
    generate_lexical_familiarization,
)
from register_feature_family.schemas import Split

V01_DATASET_FINGERPRINT = (
    "2486137265534c4bf24b0951877e48957f41f25c71f3c06a083f93b735c1e54f"
)
V01_CONFIRMATORY_SHA256 = (
    "fb723e5c6584e0987cb8a6c8574c2623777bc47cdd91de1390fadd42a4630877"
)
V01_FAMILIARIZATION_SHA256 = (
    "4f8a212f3585c8af88154952d7f112e33bd6de1de674bd851ce8088f6d2f8f0a"
)


def _canonical_json(value: object) -> str:
    """Return deterministic compact JSON."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _write_jsonl(
    path: Path,
    records: Sequence[BaseModel],
) -> None:
    """Write deterministic newline-delimited model records."""

    lines = [
        _canonical_json(
            record.model_dump(mode="json")
        )
        for record in records
    ]

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of one file."""

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _dataset_fingerprint(
    *,
    confirmatory_hash: str,
    familiarization_hash: str,
    dataset_seed: int,
    swap_variants: bool,
) -> str:
    """Calculate the deterministic v0.2 bundle fingerprint."""

    payload = {
        "dataset_version": DATASET_VERSION_V02,
        "dataset_seed": dataset_seed,
        "swap_variants": swap_variants,
        "confirmatory_sha256": confirmatory_hash,
        "lexical_familiarization_sha256": familiarization_hash,
    }

    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def export_dataset_bundle_v02(
    output_dir: Path,
    *,
    seed: int = 0,
    swap_variants: bool = False,
) -> dict[str, object]:
    """Export deterministic dataset v0.2 files and manifest."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    confirmatory_records = (
        generate_confirmatory_records_v02(
            seed=seed
        )
    )
    familiarization_records = (
        generate_lexical_familiarization(
            swap_variants=swap_variants
        )
    )

    confirmatory_path = (
        output_dir / "confirmatory.jsonl"
    )
    familiarization_path = (
        output_dir / "lexical_familiarization.jsonl"
    )
    manifest_path = output_dir / "manifest.json"

    _write_jsonl(
        confirmatory_path,
        confirmatory_records,
    )
    _write_jsonl(
        familiarization_path,
        familiarization_records,
    )

    confirmatory_hash = _sha256_file(
        confirmatory_path
    )
    familiarization_hash = _sha256_file(
        familiarization_path
    )

    split_counts = {
        split.value: sum(
            record.split == split
            for record in confirmatory_records
        )
        for split in Split
    }

    dataset_fingerprint = _dataset_fingerprint(
        confirmatory_hash=confirmatory_hash,
        familiarization_hash=familiarization_hash,
        dataset_seed=seed,
        swap_variants=swap_variants,
    )

    files: dict[str, object] = {
        "confirmatory.jsonl": {
            "records": len(confirmatory_records),
            "sha256": confirmatory_hash,
        },
        "lexical_familiarization.jsonl": {
            "records": len(familiarization_records),
            "sha256": familiarization_hash,
        },
    }

    manifest: dict[str, object] = {
        "schema_version": "0.2",
        "dataset_version": DATASET_VERSION_V02,
        "dataset_seed": seed,
        "swap_variants": swap_variants,
        "confirmatory_records": len(
            confirmatory_records
        ),
        "familiarization_records": len(
            familiarization_records
        ),
        "split_counts": split_counts,
        "files": files,
        "dataset_fingerprint": dataset_fingerprint,
        "supersedes_dataset_version": "0.1",
        "supersedes_fingerprint": (
            V01_DATASET_FINGERPRINT
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest
