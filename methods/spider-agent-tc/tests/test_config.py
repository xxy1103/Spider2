import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

import config as config_module
from config import (
    ConfigError,
    LoadedConfig,
    _validate_main,
    _validate_secrets,
    redacted_effective_config,
    select_experiment_run_dir,
    select_tasks,
)


def load_smoke():
    return yaml.safe_load((TC_ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))


def sample_items():
    return [
        {"instance_id": "a", "db_id": "DB1", "instruction": "one"},
        {"instance_id": "b", "db_id": "DB1", "instruction": "two"},
        {"instance_id": "c", "db_id": "DB2", "instruction": "three"},
        {"instance_id": "d", "db_id": "DB2", "instruction": "four"},
    ]


def task_config(**overrides):
    value = {
        "instance_ids": [],
        "index_ranges": [],
        "databases": [],
        "sample_size": None,
        "seed": 42,
        "order": "seeded_shuffle",
    }
    value.update(overrides)
    return value


def test_new_run_uses_timestamped_directory(tmp_path):
    selected = select_experiment_run_dir(
        tmp_path / "experiment",
        resume=False,
        fingerprint="fingerprint",
        now=datetime(2026, 7, 29, 19, 30, 45),
    )

    assert selected == tmp_path / "experiment" / "20260729-193045"


def test_new_run_avoids_same_second_collision(tmp_path):
    group = tmp_path / "experiment"
    (group / "20260729-193045").mkdir(parents=True)

    selected = select_experiment_run_dir(
        group,
        resume=False,
        fingerprint="fingerprint",
        now=datetime(2026, 7, 29, 19, 30, 45),
    )

    assert selected == group / "20260729-193045-01"


def test_resume_selects_latest_matching_interrupted_run(tmp_path):
    group = tmp_path / "experiment"
    older = group / "20260729-190000"
    latest = group / "20260729-193000"
    older.mkdir(parents=True)
    latest.mkdir()
    (older / "run-manifest.json").write_text(
        json.dumps({"fingerprint": "fingerprint"}), encoding="utf-8"
    )
    (older / "run-summary.json").write_text("{}", encoding="utf-8")
    (latest / "run-manifest.json").write_text(
        json.dumps({"fingerprint": "fingerprint"}), encoding="utf-8"
    )

    selected = select_experiment_run_dir(
        group,
        resume=True,
        fingerprint="fingerprint",
        now=datetime(2026, 7, 29, 20, 0, 0),
    )

    assert selected == latest


def test_resume_does_not_reuse_completed_or_changed_run(tmp_path):
    group = tmp_path / "experiment"
    completed = group / "20260729-193000"
    changed = group / "20260729-194000"
    completed.mkdir(parents=True)
    changed.mkdir()
    (completed / "run-manifest.json").write_text(
        json.dumps({"fingerprint": "fingerprint"}), encoding="utf-8"
    )
    (completed / "run-summary.json").write_text("{}", encoding="utf-8")
    (changed / "run-manifest.json").write_text(
        json.dumps({"fingerprint": "different"}), encoding="utf-8"
    )

    selected = select_experiment_run_dir(
        group,
        resume=True,
        fingerprint="fingerprint",
        now=datetime(2026, 7, 29, 20, 0, 0),
    )

    assert selected == group / "20260729-200000"


def test_smoke_config_schema_is_valid():
    raw = load_smoke()
    raw["model"]["name"] = "test-model"
    _validate_main(raw)


def test_smoke_config_loads_real_repository_paths_and_task(monkeypatch):
    original_read_yaml = config_module._read_yaml

    def fake_read_yaml(path, label):
        if label == "Secrets file":
            return {
                "model_api": {
                    "base_url": "https://api.service.test/v1",
                    "api_key": "key-value",
                },
                "snowflake": {
                    "user": "user-value",
                    "password": "token-value",
                    "account": "RSRSBDK-YDB67606",
                    "role": "PARTICIPANT",
                    "warehouse": "COMPUTE_WH_PARTICIPANT",
                },
            }
        value = original_read_yaml(path, label)
        if label == "Configuration file":
            value["model"]["name"] = "test-model"
        return value

    monkeypatch.setattr(config_module, "_read_yaml", fake_read_yaml)
    loaded = config_module.load_config(TC_ROOT / "configs" / "smoke.yaml")
    assert [item["instance_id"] for item in loaded.selected_items] == ["sf_bq011"]
    assert loaded.paths["databases"].is_dir()
    assert loaded.paths["documents"].is_dir()
    assert len(loaded.fingerprint) == 64


def test_unknown_config_field_is_rejected():
    raw = load_smoke()
    raw["model"]["name"] = "test-model"
    raw["model"]["typo"] = 123
    with pytest.raises(ConfigError, match="Unknown field"):
        _validate_main(raw)


def test_mock_config_schema_and_model_only_secrets_are_valid():
    raw = yaml.safe_load((TC_ROOT / "configs" / "mock.yaml").read_text(encoding="utf-8"))
    raw["model"]["name"] = "test-model"
    _validate_main(raw)
    _validate_secrets(
        {
            "model_api": {
                "base_url": "https://api.service.test/v1",
                "api_key": "key-value",
            }
        },
        "mock",
    )


def test_mock_mode_rejects_live_snowflake_preflight():
    raw = yaml.safe_load((TC_ROOT / "configs" / "mock.yaml").read_text(encoding="utf-8"))
    raw["model"]["name"] = "test-model"
    raw["preflight"]["check_snowflake"] = True
    with pytest.raises(ConfigError, match="must be false"):
        _validate_main(raw)


@pytest.mark.parametrize(
    "tasks,expected",
    [
        (task_config(instance_ids=["b"]), ["b"]),
        (task_config(index_ranges=["1-2"]), ["b", "c"]),
        (task_config(databases=["DB2"]), ["c", "d"]),
        (
            task_config(instance_ids=["b", "c"], index_ranges=["0-1"], databases=["DB1"]),
            ["b"],
        ),
    ],
)
def test_task_filters(tasks, expected):
    selected = select_tasks(sample_items(), tasks)
    assert sorted(item["instance_id"] for item in selected) == expected


def test_seeded_sampling_and_shuffle_are_reproducible():
    tasks = task_config(sample_size=3, seed=7)
    first = [item["instance_id"] for item in select_tasks(sample_items(), tasks)]
    second = [item["instance_id"] for item in select_tasks(sample_items(), tasks)]
    assert first == second
    assert len(first) == 3


@pytest.mark.parametrize(
    "tasks,message",
    [
        (task_config(instance_ids=["missing"]), "Unknown tasks.instance_ids"),
        (task_config(index_ranges=["4"]), "outside"),
        (task_config(index_ranges=["bad"]), "Invalid"),
        (task_config(databases=["missing"]), "Unknown tasks.databases"),
        (task_config(sample_size=5), "exceeds"),
        (
            task_config(instance_ids=["a"], databases=["DB2"]),
            "selected zero items",
        ),
    ],
)
def test_invalid_task_selection_is_rejected(tasks, message):
    with pytest.raises(ConfigError, match=message):
        select_tasks(sample_items(), tasks)


def test_effective_config_redacts_secrets(tmp_path):
    raw = load_smoke()
    loaded = LoadedConfig(
        config_path=tmp_path / "config.yaml",
        repo_root=tmp_path,
        raw=raw,
        secrets={
            "model_api": {"base_url": "https://api.example/v1", "api_key": "key-value"},
            "snowflake": {
                "user": "user-value",
                "password": "token-value",
                "account": "account",
                "role": "role",
                "warehouse": "warehouse",
            },
        },
        paths={key: tmp_path / value for key, value in raw["paths"].items()},
        selected_items=sample_items()[:1],
        fingerprint="fingerprint",
    )
    snapshot = redacted_effective_config(loaded, resolved_port=5010)
    rendered = yaml.safe_dump(snapshot)
    assert "key-value" not in rendered
    assert "user-value" not in rendered
    assert "token-value" not in rendered
    assert snapshot["server"]["resolved_port"] == 5010


def test_mock_effective_config_does_not_require_snowflake_secrets(tmp_path):
    raw = yaml.safe_load((TC_ROOT / "configs" / "mock.yaml").read_text(encoding="utf-8"))
    loaded = LoadedConfig(
        config_path=tmp_path / "config.yaml",
        repo_root=tmp_path,
        raw=raw,
        secrets={
            "model_api": {"base_url": "https://api.example/v1", "api_key": "key-value"}
        },
        paths={key: tmp_path / value for key, value in raw["paths"].items()},
        selected_items=sample_items()[:1],
        fingerprint="fingerprint",
    )
    snapshot = redacted_effective_config(loaded)
    assert "snowflake" not in snapshot["secrets"]
    assert snapshot["tools"]["snowflake"]["mode"] == "mock"
