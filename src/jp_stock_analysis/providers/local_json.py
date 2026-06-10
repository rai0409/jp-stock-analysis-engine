"""Minimal local JSON helpers. No network access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jp_stock_analysis.errors import DataValidationError


def read_json(path: str | Path) -> Any:
    """Read a local JSON file."""
    json_path = Path(path)
    if not json_path.exists():
        raise DataValidationError(f"JSON file not found: {json_path}")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid JSON in {json_path}: {exc}") from exc


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Write data to a local JSON file (UTF-8, non-ASCII preserved)."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8"
    )
    return json_path
