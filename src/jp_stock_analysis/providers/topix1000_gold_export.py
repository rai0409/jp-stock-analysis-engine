"""Import adapter for future topix1000 gold export files.

This adapter is deliberately local-file only. It reads deterministic synthetic
or future gold export files from a directory, validates the repository-owned
contract schemas at the required-field level, and returns pandas DataFrames for
downstream analysis experiments.

No network, database, API credentials, broker workflow, or execution logic is
involved.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from jp_stock_analysis.errors import ProviderError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_DIR = _REPO_ROOT / "schemas"


@dataclass(frozen=True)
class GoldDatasetSpec:
    """Static metadata for one topix1000 gold export dataset."""

    name: str
    filename: str
    schema_filename: str


DATASET_SPECS: dict[str, GoldDatasetSpec] = {
    "disclosure_events": GoldDatasetSpec(
        name="disclosure_events",
        filename="disclosure_events.jsonl",
        schema_filename="disclosure_event.schema.json",
    ),
    "price_bars": GoldDatasetSpec(
        name="price_bars",
        filename="price_bars.jsonl",
        schema_filename="price_bar.schema.json",
    ),
    "security_master": GoldDatasetSpec(
        name="security_master",
        filename="security_master.jsonl",
        schema_filename="security_master.schema.json",
    ),
    "point_in_time_features": GoldDatasetSpec(
        name="point_in_time_features",
        filename="point_in_time_features.jsonl",
        schema_filename="point_in_time_feature.schema.json",
    ),
    "signal_outputs": GoldDatasetSpec(
        name="signal_outputs",
        filename="signal_outputs.jsonl",
        schema_filename="signal_output.schema.json",
    ),
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid JSON in topix1000 gold export file {path}: {exc}") from exc


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read either JSONL records or a JSON array of objects."""
    if not path.is_file():
        raise ProviderError(f"topix1000 gold export file not found: {path}")

    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"invalid JSONL record in topix1000 gold export file "
                    f"{path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ProviderError(
                    f"topix1000 gold export record must be an object: {path}:{line_number}"
                )
            records.append(payload)
        return records

    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ProviderError(f"topix1000 gold export JSON file must contain an array: {path}")
    for position, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ProviderError(
                f"topix1000 gold export array item must be an object: {path}[{position}]"
            )
    return payload


def _load_required_fields(schema_filename: str) -> set[str]:
    schema_path = _SCHEMA_DIR / schema_filename
    if not schema_path.is_file():
        raise ProviderError(f"data contract schema not found: {schema_path}")
    schema = _read_json(schema_path)
    required = schema.get("required")
    if not isinstance(required, list) or not all(isinstance(field, str) for field in required):
        raise ProviderError(f"data contract schema has invalid required list: {schema_path}")
    return set(required)


def _parse_timestamp(value: Any, *, dataset: str, field: str, row_number: int) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProviderError(
            f"{dataset} row {row_number}: {field} must be a non-empty ISO timestamp"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderError(
            f"{dataset} row {row_number}: {field} must be an ISO timestamp: {value}"
        ) from exc


def _require_fields(
    record: dict[str, Any], required: Iterable[str], dataset: str, row_number: int
) -> None:
    missing = sorted(field for field in required if field not in record)
    if missing:
        raise ProviderError(f"{dataset} row {row_number}: missing required fields: {missing}")


def _validate_point_in_time_controls(
    dataset: str,
    record: dict[str, Any],
    row_number: int,
    *,
    strict_leakage: bool,
) -> None:
    if dataset == "disclosure_events":
        disclosed_at = _parse_timestamp(
            record.get("disclosed_at"),
            dataset=dataset,
            field="disclosed_at",
            row_number=row_number,
        )
        available_from = _parse_timestamp(
            record.get("available_from_timestamp"),
            dataset=dataset,
            field="available_from_timestamp",
            row_number=row_number,
        )
        if available_from < disclosed_at:
            raise ProviderError(
                f"{dataset} row {row_number}: available_from_timestamp precedes disclosed_at"
            )
        return

    if dataset == "price_bars":
        _parse_timestamp(
            record.get("available_from_timestamp"),
            dataset=dataset,
            field="available_from_timestamp",
            row_number=row_number,
        )
        return

    if dataset == "security_master":
        effective_from = _parse_timestamp(
            record.get("effective_from"),
            dataset=dataset,
            field="effective_from",
            row_number=row_number,
        )
        if record.get("effective_to") is not None:
            effective_to = _parse_timestamp(
                record.get("effective_to"),
                dataset=dataset,
                field="effective_to",
                row_number=row_number,
            )
            if effective_to <= effective_from:
                raise ProviderError(
                    f"{dataset} row {row_number}: effective_to must be after effective_from"
                )
        return

    if dataset == "point_in_time_features":
        available_from = _parse_timestamp(
            record.get("available_from_timestamp"),
            dataset=dataset,
            field="available_from_timestamp",
            row_number=row_number,
        )
        asof = _parse_timestamp(
            record.get("asof_timestamp"),
            dataset=dataset,
            field="asof_timestamp",
            row_number=row_number,
        )
        cutoff = _parse_timestamp(
            record.get("feature_cutoff_timestamp"),
            dataset=dataset,
            field="feature_cutoff_timestamp",
            row_number=row_number,
        )
        if cutoff > asof:
            raise ProviderError(
                f"{dataset} row {row_number}: feature_cutoff_timestamp is after asof_timestamp"
            )
        if available_from > asof:
            raise ProviderError(
                f"{dataset} row {row_number}: available_from_timestamp is after asof_timestamp"
            )
        if strict_leakage and (
            record.get("leakage_safe_flag") is not True or record.get("leakage_status") != "pass"
        ):
            raise ProviderError(
                f"{dataset} row {row_number}: leakage controls must be pass and safe"
            )
        return

    if dataset == "signal_outputs":
        asof = _parse_timestamp(
            record.get("asof_timestamp"),
            dataset=dataset,
            field="asof_timestamp",
            row_number=row_number,
        )
        cutoff = _parse_timestamp(
            record.get("feature_cutoff_timestamp"),
            dataset=dataset,
            field="feature_cutoff_timestamp",
            row_number=row_number,
        )
        label_start = _parse_timestamp(
            record.get("label_start_timestamp"),
            dataset=dataset,
            field="label_start_timestamp",
            row_number=row_number,
        )
        label_end = _parse_timestamp(
            record.get("label_end_timestamp"),
            dataset=dataset,
            field="label_end_timestamp",
            row_number=row_number,
        )
        if cutoff > asof:
            raise ProviderError(
                f"{dataset} row {row_number}: feature_cutoff_timestamp is after asof_timestamp"
            )
        if label_end <= label_start:
            raise ProviderError(
                f"{dataset} row {row_number}: label_end_timestamp must be after "
                "label_start_timestamp"
            )
        if label_start < asof:
            raise ProviderError(
                f"{dataset} row {row_number}: label_start_timestamp precedes asof_timestamp"
            )
        if strict_leakage and record.get("leakage_safe_flag") is not True:
            raise ProviderError(f"{dataset} row {row_number}: leakage_safe_flag must be true")
        if record.get("research_only_flag") is not True:
            raise ProviderError(f"{dataset} row {row_number}: research_only_flag must be true")


class Topix1000GoldExportProvider:
    """Read local topix1000 gold export files with contract validation."""

    def __init__(self, export_dir: str | Path, *, strict_leakage: bool = True) -> None:
        self.export_dir = Path(export_dir)
        self.strict_leakage = strict_leakage
        if not self.export_dir.is_dir():
            raise ProviderError(f"topix1000 gold export directory not found: {self.export_dir}")

    def load_dataset(self, dataset: str, filename: str | None = None) -> pd.DataFrame:
        """Load one dataset as a DataFrame after required-field and PIT checks."""
        if dataset not in DATASET_SPECS:
            raise ProviderError(f"unknown topix1000 gold export dataset: {dataset}")

        spec = DATASET_SPECS[dataset]
        path = self.export_dir / (filename or spec.filename)
        records = _read_records(path)
        required = _load_required_fields(spec.schema_filename)
        for row_number, record in enumerate(records, start=1):
            _require_fields(record, required, dataset, row_number)
            _validate_point_in_time_controls(
                dataset, record, row_number, strict_leakage=self.strict_leakage
            )
        return pd.DataFrame(records, dtype=object)

    def load_disclosure_events(self) -> pd.DataFrame:
        return self.load_dataset("disclosure_events")

    def load_price_bars(self) -> pd.DataFrame:
        return self.load_dataset("price_bars")

    def load_security_master(self) -> pd.DataFrame:
        return self.load_dataset("security_master")

    def load_point_in_time_features(self) -> pd.DataFrame:
        return self.load_dataset("point_in_time_features")

    def load_signal_outputs(self) -> pd.DataFrame:
        return self.load_dataset("signal_outputs")
