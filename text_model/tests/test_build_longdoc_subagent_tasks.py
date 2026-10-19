from __future__ import annotations

from collections import Counter

from text_model.scripts.data.build_longdoc_subagent_tasks import (
    DEFAULT_TAXONOMY,
    DEFAULT_TRAIN,
    build,
)


def test_v361_task_pool_has_locked_counts(tmp_path):
    manifest = build(DEFAULT_TAXONOMY, DEFAULT_TRAIN, tmp_path)
    assert manifest["generation_transport"] == "codex_multi_agent_luna"
    assert manifest["http_api_used"] is False
    assert manifest["candidate_tasks"]["rows"] == 1225
    assert manifest["candidate_tasks"]["split"] == {
        "dev": 300,
        "stress": 150,
        "train": 775,
    }
    assert manifest["final_slots"]["rows"] == 980
    assert manifest["final_slots"]["split"] == {
        "dev": 240,
        "stress": 120,
        "train": 620,
    }
    assert manifest["smoke_tasks"]["rows"] == 40
    assert manifest["smoke_tasks"]["generation_kind"] == {
        "fraud": 20,
        "normal": 20,
    }
    assert manifest["smoke_tasks"]["length_bucket"] == {
        "wp_1025_1600": 10,
        "wp_1601_2600": 10,
        "wp_385_512": 10,
        "wp_513_1024": 10,
    }
    assert manifest["smoke_tasks"]["cross_window_evidence"] == {
        "False": 36,
        "True": 4,
    }


def test_v361_train_slots_are_balanced_and_cover_39_children(tmp_path):
    build(DEFAULT_TAXONOMY, DEFAULT_TRAIN, tmp_path)
    import json

    rows = [
        json.loads(line)
        for line in (tmp_path / "final_slots_980.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train = [row for row in rows if row["split"] == "train"]
    assert Counter(row["generation_kind"] for row in train) == {
        "fraud": 310,
        "normal": 310,
    }
    assert Counter(row["length_bucket"] for row in train) == {
        "wp_385_512": 155,
        "wp_513_1024": 248,
        "wp_1025_1600": 155,
        "wp_1601_2600": 62,
    }
    assert len({row["subcategory_code"] for row in train}) == 39
    fraud = [row for row in train if row["generation_kind"] == "fraud"]
    assert sum(bool(row["cross_window_evidence"]) for row in fraud) == 62
    cross = [row for row in fraud if row["cross_window_evidence"]]
    assert all(row["length_bucket"] != "wp_385_512" for row in cross)
    assert all(row["evidence_position_bin"] in {"Q3", "Q4"} for row in cross)
