import json

import pytest
import yaml

from agent.schema_router_config import (
    SchemaRouterConfigError,
    bind_schema_router_task_set,
    load_schema_router_config,
    prepare_schema_router_run,
)


def _config():
    return {
        "experiment": {
            "name": "router-test",
            "results_root": "methods/spider-agent-tc/results",
            "resume": True,
        },
        "secrets_file": "router-secrets.yaml",
        "paths": {
            "input_file": "router-tasks.jsonl",
            "databases": "router-databases",
            "documents": "router-documents",
            "official_sql_dir": "router-sql",
            "router_prompt": "router-prompt.txt",
        },
        "schema_router": {
            "model": {
                "name": "mock-model",
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 1000,
                "request_timeout_seconds": 30,
                "retry": {
                    "max_attempts": 1,
                    "initial_delay_seconds": 0,
                    "backoff_multiplier": 1,
                    "max_delay_seconds": 0,
                },
            },
            "max_rounds": 6,
            "max_tool_calls": 24,
            "num_threads": 1,
            "sample_rows": 1,
            "max_sample_chars": 1000,
        },
        "evaluation": {
            "rollouts": 1,
            "thresholds": {
                "physical_task_full_coverage": 0.97,
                "physical_micro_recall": 0.99,
                "invalid_references": 0,
            },
        },
    }


def _materialize(monkeypatch, tmp_path, config_value):
    (tmp_path / "router-databases").mkdir()
    (tmp_path / "router-documents").mkdir()
    (tmp_path / "router-sql").mkdir()
    (tmp_path / "router-prompt.txt").write_text("router", encoding="utf-8")
    (tmp_path / "router-tasks.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "sf_test",
                "instruction": "test",
                "db_id": "DB",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "router-secrets.yaml").write_text(
        yaml.safe_dump(
            {
                "model_api": {
                    "base_url": "https://model.test/v1",
                    "api_key": "secret-value",
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "router.yaml"
    config_path.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    import agent.schema_router_config as module

    original_path = module.Path
    monkeypatch.setattr(
        module,
        "_resolve",
        lambda repo_root, value, location: (
            original_path(value).resolve()
            if original_path(value).is_absolute()
            else (tmp_path / str(value)).resolve()
        ),
    )
    return config_path


def test_router_config_loads_without_snowflake_secrets(monkeypatch, tmp_path):
    path = _materialize(monkeypatch, tmp_path, _config())
    loaded = load_schema_router_config(path)
    assert loaded.raw["evaluation"]["rollouts"] == 1
    assert set(loaded.secrets) == {"model_api"}
    assert loaded.paths["official_sql_dir"] == (tmp_path / "router-sql").resolve()


def test_router_config_rejects_unknown_fields(monkeypatch, tmp_path):
    value = _config()
    value["schema_router"]["unexpected"] = True
    path = _materialize(monkeypatch, tmp_path, value)
    with pytest.raises(SchemaRouterConfigError, match="Unknown field"):
        load_schema_router_config(path)


def test_router_thinking_fields_must_be_configured_as_a_pair(monkeypatch, tmp_path):
    value = _config()
    value["schema_router"]["model"]["provider"] = "deepseek"
    path = _materialize(monkeypatch, tmp_path, value)
    with pytest.raises(SchemaRouterConfigError, match="configured together"):
        load_schema_router_config(path)


def test_router_rejects_provider_specific_thinking_level(monkeypatch, tmp_path):
    value = _config()
    value["schema_router"]["model"].update(
        provider="gemini", thinking_level="xhigh"
    )
    path = _materialize(monkeypatch, tmp_path, value)
    with pytest.raises(SchemaRouterConfigError, match="not supported for gemini"):
        load_schema_router_config(path)


def test_router_config_rejects_legacy_split_fields(monkeypatch, tmp_path):
    value = _config()
    value["evaluation"]["phase"] = "development"
    path = _materialize(monkeypatch, tmp_path, value)
    with pytest.raises(SchemaRouterConfigError, match="Legacy.*phase.*rollouts"):
        load_schema_router_config(path)


def test_router_config_rejects_non_positive_rollouts(monkeypatch, tmp_path):
    value = _config()
    value["evaluation"]["rollouts"] = 0
    path = _materialize(monkeypatch, tmp_path, value)
    with pytest.raises(SchemaRouterConfigError, match="rollouts must be at least 1"):
        load_schema_router_config(path)


def test_prompt_content_changes_fingerprint(monkeypatch, tmp_path):
    path = _materialize(monkeypatch, tmp_path, _config())
    first = load_schema_router_config(path)
    (tmp_path / "router-prompt.txt").write_text("router v2", encoding="utf-8")
    second = load_schema_router_config(path)

    assert first.fingerprint != second.fingerprint


def test_protocol_version_changes_fingerprint(monkeypatch, tmp_path):
    path = _materialize(monkeypatch, tmp_path, _config())
    first = load_schema_router_config(path)
    import agent.schema_router_config as module

    monkeypatch.setattr(module, "SCHEMA_ROUTER_PROTOCOL_VERSION", 999)
    second = load_schema_router_config(path)

    assert first.fingerprint != second.fingerprint


def test_task_set_changes_fingerprint(monkeypatch, tmp_path):
    path = _materialize(monkeypatch, tmp_path, _config())
    config = load_schema_router_config(path)

    first = bind_schema_router_task_set(config, {"instance_ids": ["a"]})
    second = bind_schema_router_task_set(config, {"instance_ids": ["a", "b"]})

    assert first.fingerprint != second.fingerprint


def test_changed_prompt_starts_a_new_run_directory(monkeypatch, tmp_path):
    value = _config()
    value["experiment"]["results_root"] = str(tmp_path / "results")
    path = _materialize(monkeypatch, tmp_path, value)
    first = prepare_schema_router_run(load_schema_router_config(path))
    (tmp_path / "router-prompt.txt").write_text("router v2", encoding="utf-8")
    second = prepare_schema_router_run(load_schema_router_config(path))

    assert first.experiment_dir != second.experiment_dir
