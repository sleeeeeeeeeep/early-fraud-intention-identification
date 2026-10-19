"""Official normal + 38 fraud subcategory indexing for text-model auxiliaries."""

from __future__ import annotations

from typing import Any, Callable


OFFICIAL_SUBCATEGORY_CODES = (
    "00_0000",
    "01_0101", "01_0102", "01_0103",
    "02_0201", "02_0202",
    "03_0301",
    "04_0401", "04_0402", "04_0403", "04_0404",
    "05_0501",
    "06_0601", "06_0602", "06_0603",
    "07_0701", "07_0702", "07_0703",
    "08_0801", "08_0802", "08_0803", "08_0804",
    "09_0901", "09_0902", "09_0903",
    "10_1001", "10_1002", "10_1003", "10_1004",
    "11_1101", "11_1102", "11_1103",
    "12_1201", "12_1202", "12_1203", "12_1204", "12_1205", "12_1206", "12_1207",
)
OFFICIAL_SUBCATEGORY_TO_INDEX = {code: idx for idx, code in enumerate(OFFICIAL_SUBCATEGORY_CODES)}
NUM_OFFICIAL_SUBCATEGORIES = len(OFFICIAL_SUBCATEGORY_CODES)
OFFICIAL_PARENT_TO_INDICES = {
    parent: tuple(
        idx
        for idx, code in enumerate(OFFICIAL_SUBCATEGORY_CODES)
        if (0 if code == "00_0000" else int(code[:2])) == parent
    )
    for parent in range(13)
}


def official_parent_of_index(index: int) -> int:
    code = OFFICIAL_SUBCATEGORY_CODES[index]
    return 0 if code == "00_0000" else int(code[:2])


def sibling_local_target(parent: int, global_index: int) -> int:
    siblings = OFFICIAL_PARENT_TO_INDICES[parent]
    if global_index not in siblings:
        raise ValueError(
            f"official subcategory index {global_index} is not a child of parent {parent}"
        )
    return siblings.index(global_index)


def official_subcategory_code_of(row: dict[str, Any], label_of: Callable[[dict[str, Any]], int]) -> str:
    label = label_of(row)
    raw = str(row.get("subcategory_code") or row.get("classification_raw") or "").strip()
    if label == 0 and (not raw or "13_1301" in raw):
        return "00_0000"
    if not raw:
        return ""
    for code in OFFICIAL_SUBCATEGORY_CODES:
        if code in raw:
            return code
    return ""


def official_subcategory_index_of(row: dict[str, Any], label_of: Callable[[dict[str, Any]], int]) -> int:
    code = official_subcategory_code_of(row, label_of)
    if not code:
        return -100
    label = label_of(row)
    expected_label = 0 if code == "00_0000" else int(code[:2])
    if label != expected_label:
        raise ValueError(f"official subcategory/parent mismatch: code={code!r}, label_13={label}")
    return OFFICIAL_SUBCATEGORY_TO_INDEX[code]
