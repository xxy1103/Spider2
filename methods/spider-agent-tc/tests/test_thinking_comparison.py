import json
from pathlib import Path

import yaml
import pytest
from types import SimpleNamespace

import thinking_comparison
from thinking_comparison import (
    LEVELS,
    _build_child_raw,
    _build_final_scores,
    build_comparison_summary,
    build_fixed_task_set,
    _find_completed_child,
    _probe_model,
)


def test_fixed_task_set_samples_50_of_120_deterministically(tmp_path):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    input_path = tmp_path / "input.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    rows = []
    for index in range(120):
        instance_id = f"sf_{index:03d}"
        (sql_dir / f"{instance_id}.sql").write_text("select 1", encoding="utf-8")
        rows.append(json.dumps({"instance_id": instance_id}))
    text = "\n".join(rows) + "\n"
    input_path.write_text(text, encoding="utf-8")
    eval_path.write_text(text, encoding="utf-8")
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump({"paths": {"input_file": "input.jsonl"}}),
        encoding="utf-8",
    )
    raw = {
        "repo_root": str(tmp_path),
        "base_config": str(base_path),
        "official_sql_dir": str(sql_dir),
        "evaluation_standard": str(eval_path),
        "sampling": {"sample_size": 50, "seed": 42},
    }

    first = build_fixed_task_set(raw)
    second = build_fixed_task_set(raw)

    assert first == second
    assert first["source_task_count"] == 120
    assert len(first["sampled_instance_ids"]) == 50
    assert len(set(first["sampled_instance_ids"])) == 50


def test_child_conditions_change_only_controlled_experiment_fields(tmp_path):
    base = {
        "experiment": {},
        "tasks": {},
        "model": {"temperature": 0.2},
        "agent": {"rollout_number": 3},
        "schema_router": {
            "enabled": True,
            "model": {"temperature": 0.2},
            "integration": {"mode": "strict", "failure_policy": "fail_task"},
        },
        "tools": {"sql": {"mode": "mock"}},
        "preflight": {},
        "auto_evaluate": {"enabled": True, "timeout": 9, "max_workers": 2},
    }
    raw = {
        "repo_root": str(tmp_path),
        "sampling": {"seed": 42},
    }
    task_set = {"sampled_instance_ids": [f"task-{index}" for index in range(50)]}
    children = {
        level: _build_child_raw(
            base_raw=base,
            raw=raw,
            parent_dir=tmp_path / "run",
            task_set=task_set,
            level=level,
        )
        for level in LEVELS
    }

    for level, child in children.items():
        assert child["model"]["thinking_level"] == level
        assert child["schema_router"]["model"]["thinking_level"] == level
        assert child["model"]["temperature"] == 0
        assert child["schema_router"]["model"]["temperature"] == 0
        assert child["agent"]["rollout_number"] == 1
        assert child["tools"]["sql"]["mode"] == "live"
        assert child["tasks"]["instance_ids"] == task_set["sampled_instance_ids"]


def test_final_scores_keep_missing_submission_in_fifty_task_denominator(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "a.sql").write_text("select 1", encoding="utf-8")
    (submission / "b.sql").write_text("select 2", encoding="utf-8")

    result = _build_final_scores(
        ["a", "b", "c"],
        [
            {"instance_id": "a", "score": 1, "error_info": None},
            {"instance_id": "b", "score": 0, "error_info": "wrong"},
        ],
        submission,
    )

    assert result["submitted_tasks"] == 2
    assert result["evaluated_tasks"] == 2
    assert result["correct_tasks"] == 1
    assert result["completion_rate"] == 2 / 3
    assert result["submitted_answer_accuracy"] == 1 / 2
    assert result["end_to_end_accuracy"] == 1 / 3
    assert result["tasks"][2]["score"] == 0


def test_comparison_summary_counts_pairwise_flips_against_high():
    conditions = {}
    final_values = {
        "none": [1, 0],
        "low": [1, 1],
        "high": [0, 1],
        "max": [0, 0],
    }
    router_values = {
        "none": [True, False],
        "low": [True, True],
        "high": [False, True],
        "max": [False, False],
    }
    for level in LEVELS:
        conditions[level] = {
            "final": {
                "tasks": [
                    {"instance_id": key, "score": score}
                    for key, score in zip(("a", "b"), final_values[level])
                ]
            },
            "router": {
                "tasks": [
                    {"instance_id": key, "physical_full_coverage": score}
                    for key, score in zip(("a", "b"), router_values[level])
                ]
            },
        }

    summary = build_comparison_summary(
        task_set={"sampled_instance_ids_sorted": ["a", "b"]},
        conditions=conditions,
        probes={},
    )

    assert summary["pairwise_vs_high"]["none"] == {
        "final_wrong_to_right_vs_high": 1,
        "final_right_to_wrong_vs_high": 1,
        "router_miss_to_full_vs_high": 1,
        "router_full_to_miss_vs_high": 1,
    }


def test_probe_rejects_model_fallback(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(
                model="fallback-model",
                system_fingerprint="fingerprint",
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=1,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(thinking_comparison, "OpenAI", FakeOpenAI)
    config = SimpleNamespace(
        raw={
            "model": {
                "name": "deepseek-v4-flash",
                "provider": "deepseek",
                "thinking_level": "low",
                "request_timeout_seconds": 10,
            }
        },
        secrets={"model_api": {"base_url": "https://example.test", "api_key": "x"}},
    )

    with pytest.raises(RuntimeError, match="Model fallback detected"):
        _probe_model(config, "low")


def test_find_completed_child_requires_matching_fingerprint(tmp_path):
    group = tmp_path / "group"
    matching = group / "20260806-120000"
    different = group / "20260806-130000"
    matching.mkdir(parents=True)
    different.mkdir()
    (matching / "run-manifest.json").write_text(
        json.dumps({"fingerprint": "same"}), encoding="utf-8"
    )
    (matching / "run-summary.json").write_text("{}", encoding="utf-8")
    (different / "run-manifest.json").write_text(
        json.dumps({"fingerprint": "different"}), encoding="utf-8"
    )
    (different / "run-summary.json").write_text("{}", encoding="utf-8")
    config = SimpleNamespace(experiment_group_dir=group, fingerprint="same")

    assert _find_completed_child(config) == matching
