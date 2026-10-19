"""Shared strict parsing helpers for labeled telefraud datasets."""

from __future__ import annotations

import math
from typing import Any, Mapping


class DataContractError(ValueError):
    """Raised when a row violates a required dataset contract."""


def _location(context: str) -> str:
    return f"{context}: " if context else ""


def strict_int_label(value: Any, *, field: str, context: str = "") -> int:
    """Parse an integer label in the closed interval 0..12 without fallback."""
    prefix = _location(context)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise DataContractError(f"{prefix}{field} is missing or empty")
    if isinstance(value, bool):
        raise DataContractError(f"{prefix}{field} must be an integer in 0..12, got boolean {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{prefix}{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise DataContractError(f"{prefix}{field} is not an integer: {value!r}")
    label = int(number)
    if not 0 <= label <= 12:
        raise DataContractError(f"{prefix}{field} is out of range 0..12: {value!r}")
    return label


def strict_label13_of(row: Mapping[str, Any], *, context: str = "") -> int:
    """Read label_13 (preferred) or label and reject missing/conflicting values."""
    has_label13 = "label_13" in row
    has_label = "label" in row
    if not has_label13 and not has_label:
        raise DataContractError(f"{_location(context)}missing required label_13 or label")

    label13 = strict_int_label(row.get("label_13"), field="label_13", context=context) if has_label13 else None
    label = strict_int_label(row.get("label"), field="label", context=context) if has_label else None
    if label13 is not None and label is not None and label13 != label:
        raise DataContractError(
            f"{_location(context)}label_13 ({label13}) conflicts with label ({label})"
        )
    return label13 if label13 is not None else label  # type: ignore[return-value]


def strict_prediction_label_of(row: Mapping[str, Any], field: str, *, context: str = "") -> int:
    """Strictly parse an explicitly named true/prediction label field."""
    if field not in row:
        raise DataContractError(f"{_location(context)}missing required {field}")
    return strict_int_label(row[field], field=field, context=context)
