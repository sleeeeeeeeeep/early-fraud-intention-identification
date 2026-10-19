from __future__ import annotations

import json
import os
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _as_number(value: int | float | None) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _status_lock(path: Path) -> Path:
    return path.parent / "status.lock"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def update_progress(
    *,
    status_file: str | Path | None = None,
    phase: str,
    phase_current: str | None = None,
    phase_done: int | float | None = None,
    phase_total: int | float | None = None,
    epoch_current: int | float | None = None,
    epoch_total: int | float | None = None,
    extra: dict[str, Any] | None = None,
    force: bool = False,
    min_interval_seconds: int | float | None = 180,
) -> dict[str, Any]:
    """Update the standard subrun status.json with lightweight phase progress."""
    target_value = status_file or os.environ.get("STANDARD_STATUS_FILE")
    if not target_value:
        return {}
    target = Path(target_value)
    lock = _status_lock(target)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            payload = _read_json(target)
            if not force and min_interval_seconds:
                last = _parse_iso(payload.get("progress_updated_at"))
                now = datetime.now(timezone.utc).astimezone()
                if last and (now - last).total_seconds() < float(min_interval_seconds):
                    return payload
            else:
                now = datetime.now(timezone.utc).astimezone()
            now_text = now.strftime("%Y-%m-%dT%H:%M:%S%z")
            payload.update(
                {
                    "phase": phase,
                    "phase_current": phase_current,
                    "phase_done": _as_number(phase_done),
                    "phase_total": _as_number(phase_total),
                    "epoch_current": _as_number(epoch_current),
                    "epoch_total": _as_number(epoch_total),
                    "progress_updated_at": now_text,
                    "updated_at": now_text,
                }
            )
            if extra:
                payload.update(extra)
            _write_json_atomic(target, payload)
            return payload
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def update_train_progress(
    *,
    epoch_current: int | float | None,
    epoch_total: int | float | None,
    step_current: int | float | None = None,
    step_total: int | float | None = None,
    status_file: str | Path | None = None,
    last_epoch_seconds: int | float | None = None,
    force: bool = False,
    min_interval_seconds: int | float | None = 180,
) -> dict[str, Any]:
    extra = {}
    if _as_number(last_epoch_seconds) is not None:
        extra["last_epoch_seconds"] = _as_number(last_epoch_seconds)
    return update_progress(
        status_file=status_file,
        phase="train",
        phase_current="step" if step_current is not None else "epoch",
        phase_done=step_current if step_current is not None else epoch_current,
        phase_total=step_total if step_total is not None else epoch_total,
        epoch_current=epoch_current,
        epoch_total=epoch_total,
        extra=extra or None,
        force=force,
        min_interval_seconds=min_interval_seconds,
    )


def update_eval_progress(
    *,
    dataset: str,
    done: int | float | None,
    total: int | float | None,
    status_file: str | Path | None = None,
    force: bool = True,
) -> dict[str, Any]:
    return update_progress(
        status_file=status_file,
        phase="eval",
        phase_current=dataset,
        phase_done=done,
        phase_total=total,
        force=force,
        min_interval_seconds=0,
    )
