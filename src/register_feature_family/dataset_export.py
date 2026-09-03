import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from register_feature_family.experimental_generator import (
    generate_confirmatory_records,
    generate_lexical_familiarization,
)
from register_feature_family.schemas import Split

CONFIRMATORY_FILENAME = "confirmatory.jsonl"
FAMILIARIZATION_FILENAME = "lexical_familiarization.jsonl"
MANIFEST_FILENAME = "manifest.json"


def _canonical_json(value: dict[str, Any]) -> str:
    """Serialize one object deterministically."""

    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_jsonl(
    *,
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write deterministic newline-delimited JSON."""

    text = "".join(
        f"{_canonical_json(row)}\n"
        for row in rows
    )

    path.write_text(text, encoding="utf-8", newline="\n")


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


def _combined_fingerprint(
    *,
    confirmatory_sha256: str,
    familiarization_sha256: str,
    seed: int,
    swap_variants: bool,
) -> str:
    """Return a deterministic fingerprint for the complete bundle."""

    value = (
        f"confirmatory={confirmatory_sha256}\n"
        f"familiarization={familiarization_sha256}\n"
        f"seed={seed}\n"
        f"swap_variants={swap_variants}\n"
    )

    return hashlib.sha256(value.encode()).hexdigest()


def export_dataset_bundle(
    *,
    output_dir: Path,
    seed: int = 0,
    swap_variants: bool = False,
) -> dict[str, Any]:
    """Export the frozen confirmatory dataset and manifest."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    confirmatory_records = generate_confirmatory_records(
        seed=seed
    )
    familiarization_records = generate_lexical_familiarization(
        swap_variants=swap_variants
    )

    confirmatory_rows = [
        record.model_dump(mode="json")
        for record in confirmatory_records
    ]
    familiarization_rows = [
        record.model_dump(mode="json")
        for record in familiarization_records
    ]

    confirmatory_path = output_dir / CONFIRMATORY_FILENAME
    familiarization_path = output_dir / FAMILIARIZATION_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    _write_jsonl(
        path=confirmatory_path,
        rows=confirmatory_rows,
    )
    _write_jsonl(
        path=familiarization_path,
        rows=familiarization_rows,
    )

    split_counts = Counter(
        record.split
        for record in confirmatory_records
    )

    confirmatory_sha256 = _sha256_file(confirmatory_path)
    familiarization_sha256 = _sha256_file(
        familiarization_path
    )

    manifest: dict[str, Any] = {
        "schema_version": "0.1",
        "dataset_seed": seed,
        "swap_variants": swap_variants,
        "confirmatory_records": len(
            confirmatory_records
        ),
        "familiarization_records": len(
            familiarization_records
        ),
        "split_counts": {
            split.value: split_counts[split]
            for split in Split
        },
        "files": {
            CONFIRMATORY_FILENAME: {
                "sha256": confirmatory_sha256,
                "records": len(confirmatory_records),
            },
            FAMILIARIZATION_FILENAME: {
                "sha256": familiarization_sha256,
                "records": len(familiarization_records),
            },
        },
    }

    manifest["dataset_fingerprint"] = _combined_fingerprint(
        confirmatory_sha256=confirmatory_sha256,
        familiarization_sha256=familiarization_sha256,
        seed=seed,
        swap_variants=swap_variants,
    )

    manifest_path.write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
        newline="\n",
    )

    return manifest
