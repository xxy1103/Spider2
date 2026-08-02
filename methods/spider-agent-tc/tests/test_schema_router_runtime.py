import json
from types import SimpleNamespace

import pytest

from run_schema_router import (
    _build_jobs,
    _evaluation_exit_code,
    _partition_jobs_for_resume,
    _validate_or_write_task_set,
)


def test_resume_only_schedules_incomplete_rollouts(tmp_path):
    config = SimpleNamespace(experiment_dir=tmp_path)
    valid_path = tmp_path / "routing" / "task-a" / "rollout-0.json"
    valid_path.parent.mkdir(parents=True)
    valid = {
        "instance_id": "task-a",
        "rollout_idx": 0,
        "completed": True,
        "selection": {"valid": True},
    }
    valid_path.write_text(json.dumps(valid), encoding="utf-8")
    invalid_path = tmp_path / "routing" / "task-b" / "rollout-0.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text(
        json.dumps({"instance_id": "task-b", "completed": False}),
        encoding="utf-8",
    )
    jobs = [
        ({"instance_id": "task-a"}, 0),
        ({"instance_id": "task-b"}, 0),
        ({"instance_id": "task-c"}, 0),
    ]

    completed, pending = _partition_jobs_for_resume(config, jobs)

    assert completed == [valid]
    assert [(task["instance_id"], rollout) for task, rollout in pending] == [
        ("task-b", 0),
        ("task-c", 0),
    ]


def test_full_task_set_schedules_every_rollout():
    tasks = {
        "task-a": {"instance_id": "task-a"},
        "task-b": {"instance_id": "task-b"},
    }

    jobs = _build_jobs(tasks, ["task-a", "task-b"], 2)

    assert [(task["instance_id"], rollout) for task, rollout in jobs] == [
        ("task-a", 0),
        ("task-a", 1),
        ("task-b", 0),
        ("task-b", 1),
    ]


def test_task_set_manifest_is_stable_and_rejects_mismatch(tmp_path):
    config = SimpleNamespace(experiment_dir=tmp_path)
    task_set = {
        "task_count": 2,
        "instance_ids": ["task-a", "task-b"],
        "labels_sha256": "labels",
        "catalog_sha256": "catalog",
    }

    _validate_or_write_task_set(config=config, task_set=task_set)
    _validate_or_write_task_set(config=config, task_set=task_set)

    with pytest.raises(RuntimeError, match="task-set manifest differs"):
        _validate_or_write_task_set(
            config=config,
            task_set={**task_set, "instance_ids": ["task-a"]},
        )


def test_threshold_exit_codes():
    assert _evaluation_exit_code({"passed": True}) == 0
    assert _evaluation_exit_code({"passed": False}) == 2
