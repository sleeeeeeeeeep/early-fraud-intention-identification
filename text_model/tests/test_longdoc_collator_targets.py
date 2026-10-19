"""Checks that soft targets accept official subcategory codes as keys."""

from __future__ import annotations

import pytest

from data.longdoc_collator import NUM_CHILD_CLASSES, NUM_PARENT_CLASSES, _target_distribution
from scripts.inference.official_subcategories import OFFICIAL_SUBCATEGORY_TO_INDEX


def test_child_target_accepts_official_subcategory_code() -> None:
    row = {"child_target": {"kind": "hard", "probabilities": {"05_0501": 1.0}}}
    values = _target_distribution(
        row, "child_target", 0, NUM_CHILD_CLASSES, OFFICIAL_SUBCATEGORY_TO_INDEX
    )
    assert values[OFFICIAL_SUBCATEGORY_TO_INDEX["05_0501"]] == 1.0
    assert sum(values) == 1.0


def test_child_target_accepts_plain_index_key() -> None:
    row = {"child_target": {"probabilities": {"7": 0.6, "8": 0.4}}}
    values = _target_distribution(
        row, "child_target", 0, NUM_CHILD_CLASSES, OFFICIAL_SUBCATEGORY_TO_INDEX
    )
    assert (values[7], values[8]) == (0.6, 0.4)


def test_parent_target_keeps_index_semantics() -> None:
    row = {"parent_target": {"kind": "hard", "probabilities": {"4": 1.0}}}
    values = _target_distribution(row, "parent_target", 0, NUM_PARENT_CLASSES)
    assert values[4] == 1.0


def test_unknown_child_code_is_rejected() -> None:
    row = {"child_target": {"probabilities": {"99_9999": 1.0}}}
    with pytest.raises(ValueError, match="code is unknown"):
        _target_distribution(
            row, "child_target", 0, NUM_CHILD_CLASSES, OFFICIAL_SUBCATEGORY_TO_INDEX
        )


def test_missing_target_falls_back_to_one_hot() -> None:
    values = _target_distribution({}, "child_target", 3, NUM_CHILD_CLASSES)
    assert values[3] == 1.0
    assert sum(values) == 1.0
