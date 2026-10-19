from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_history_entry(
    *,
    index: int | float,
    unit: str = "epoch",
    train_loss: float | None = None,
    selection_score: float | None = None,
    val: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "index": index,
        "unit": unit,
        "timestamp": timestamp or _now_iso(),
        "epoch": index if unit == "epoch" else None,
        "train_loss": _as_float_or_none(train_loss),
        "selection_score": _as_float_or_none(selection_score),
        "val": val if isinstance(val, dict) else None,
        "extras": extras if isinstance(extras, dict) else {},
    }
    return entry


def append_epoch_history(
    output_dir: str | Path,
    *,
    index: int | float,
    train_loss: float | None = None,
    selection_score: float | None = None,
    val: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
    timestamp: str | None = None,
    history_name: str = "history.json",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    history_path = output_path / history_name
    rows = _read_history(history_path)
    entry = build_history_entry(
        index=index,
        unit="epoch",
        train_loss=train_loss,
        selection_score=selection_score,
        val=val,
        extras=extras,
        timestamp=timestamp,
    )
    rows.append(entry)
    _write_history(history_path, rows)
    return entry


def write_standard_history(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    history_name: str = "history.json",
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, float]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        raw_unit = raw.get("unit")
        raw_epoch = _as_float_or_none(raw.get("epoch"))
        raw_step = _as_float_or_none(raw.get("step"))
        index = _as_float_or_none(raw.get("index"))
        unit = raw_unit if isinstance(raw_unit, str) and raw_unit.strip() else ""
        if not unit:
            unit = "step" if raw_step is not None or (raw_epoch is not None and not raw_epoch.is_integer()) else "epoch"
        if index is None:
            index = raw_step if unit == "step" and raw_step is not None else raw_epoch
        if index is None:
            continue
        key = (unit, index)
        if key not in merged:
            merged[key] = build_history_entry(index=index, unit=unit, timestamp=raw.get("timestamp") if isinstance(raw.get("timestamp"), str) else None)
            ordered_keys.append(key)
        entry = merged[key]
        if _as_float_or_none(raw.get("loss")) is not None:
            entry["train_loss"] = _as_float_or_none(raw.get("loss"))
        if _as_float_or_none(raw.get("train_loss")) is not None:
            entry["train_loss"] = _as_float_or_none(raw.get("train_loss"))
        if _as_float_or_none(raw.get("selection_score")) is not None:
            entry["selection_score"] = _as_float_or_none(raw.get("selection_score"))
        val_payload = entry.get("val") if isinstance(entry.get("val"), dict) else {}
        for key, value in raw.items():
            if key.startswith("eval_"):
                val_payload[key] = value
        if isinstance(raw.get("val"), dict):
            val_payload = {**val_payload, **raw["val"]}
        entry["val"] = val_payload or None
        extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
        if unit == "step" and raw_epoch is not None:
            extras["epoch_progress"] = raw_epoch
        for key, value in raw.items():
            if key in {"index", "unit", "timestamp", "epoch", "step", "loss", "train_loss", "selection_score", "val"} or key.startswith("eval_"):
                continue
            extras[key] = value
        entry["extras"] = extras
    normalized = [merged[key] for key in ordered_keys]
    _write_history(Path(output_dir) / history_name, normalized)
    return normalized
