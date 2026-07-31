import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

from agent.langgraph_agent import LangGraphAgent
from agent.llm_agent import LLMAgent
from config import ConfigError, LoadedConfig
from run import find_available_port, prepare_experiment, start_server, stop_server


def test_port_falls_back_when_preferred_is_busy():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        preferred = occupied.getsockname()[1]
        selected = find_available_port("127.0.0.1", preferred)
    assert selected != preferred
    assert selected > 0


def test_stop_server_terminates_live_process():
    process = Mock()
    process.poll.return_value = None
    stop_server(process)
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)


def test_stop_server_kills_process_that_does_not_terminate():
    import subprocess

    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 5), None]
    stop_server(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_start_server_redirects_output_to_experiment_log(tmp_path, monkeypatch):
    popen = Mock()
    monkeypatch.setattr("run.subprocess.Popen", popen)
    config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        experiment_dir=tmp_path,
        raw={"server": {"host": "127.0.0.1"}},
    )

    start_server(config, 5000)

    kwargs = popen.call_args.kwargs
    assert kwargs["stderr"] == -2
    assert kwargs["stdout"].name == str(tmp_path / "run.log")
    assert kwargs["stdout"].closed is True


def runtime_config(tmp_path, fingerprint="same", resume=True):
    return LoadedConfig(
        config_path=tmp_path / "config.yaml",
        repo_root=tmp_path,
        raw={
            "experiment": {"name": "experiment", "results_root": "results", "resume": resume},
            "tasks": {"seed": 42},
        },
        secrets={
            "model_api": {"base_url": "https://api.test/v1", "api_key": "key"},
            "snowflake": {
                "user": "user",
                "password": "token",
                "account": "account",
                "role": "role",
                "warehouse": "warehouse",
            },
        },
        paths={},
        selected_items=[{"instance_id": "task"}],
        fingerprint=fingerprint,
    )


def test_prepare_experiment_allows_matching_resume(tmp_path, monkeypatch):
    config = runtime_config(tmp_path)
    monkeypatch.setattr(
        "run.redacted_effective_config",
        lambda config, resolved_port=None: {"safe": True, "port": resolved_port},
    )
    prepare_experiment(config, 5000)
    prepare_experiment(config, 5001)
    assert (config.experiment_dir / "run-manifest.json").is_file()


def test_prepare_experiment_rejects_changed_config(tmp_path, monkeypatch):
    config = runtime_config(tmp_path)
    monkeypatch.setattr(
        "run.redacted_effective_config",
        lambda config, resolved_port=None: {"safe": True},
    )
    prepare_experiment(config, 5000)
    changed = runtime_config(tmp_path, fingerprint="changed")
    with pytest.raises(ConfigError, match="configuration differs"):
        prepare_experiment(changed, 5000)


def test_model_retry_stops_after_configured_attempts(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(
        model_base_url="https://api.example/v1",
        model_api_key="secret",
        model_request_timeout=1,
        retry={
            "max_attempts": 3,
            "initial_delay_seconds": 0,
            "backoff_multiplier": 2,
            "max_delay_seconds": 0,
        },
        output_folder=str(tmp_path),
        prompt_strategy="spider-agent",
        tool_request_timeout=1,
        databases_path=str(tmp_path),
        model="model",
        temperature=0,
        top_p=1,
        max_new_tokens=10,
    )
    monkeypatch.setattr("agent.langgraph_agent.OpenAI", Mock)
    agent = LLMAgent(args)
    create = agent.model_client.chat.completions.create
    create.side_effect = RuntimeError("secret should not be printed")
    result = agent.call_llm([{"role": "user", "content": "test"}])
    assert result == "ERROR: Failed to get response after 3 attempts"
    assert create.call_count == 3
    assert "secret" not in capsys.readouterr().out


def test_model_round_logs_provider_reasoning_and_assistant_content(caplog):
    agent = object.__new__(LangGraphAgent)
    agent.args = SimpleNamespace(model_api_key="secret")
    response_message = SimpleNamespace(
        content="I will inspect the relevant table.",
        reasoning_content="The task requires one row per month.",
        tool_calls=[],
        model_extra=None,
    )
    agent._call_llm_with_retry = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=response_message)]
        )
    )
    state = {
        "messages": [],
        "item": {"instance_id": "task"},
        "conversation_history": [],
        "round_num": 0,
        "rollout_idx": 1,
        "performance": LangGraphAgent._new_performance(),
    }

    with caplog.at_level("INFO", logger="agent.langgraph_agent"):
        result = agent._call_model_node(state)

    assert "MODEL_ROUND instance=task rollout=2 round=1" in caplog.text
    assert "THINKING:\nThe task requires one row per month." in caplog.text
    assert "ASSISTANT_CONTENT:\nI will inspect the relevant table." in caplog.text
    assert result["conversation_history"][-1]["reasoning_content"] == (
        "The task requires one row per month."
    )


def test_model_round_uses_working_note_when_provider_has_no_reasoning(caplog):
    agent = object.__new__(LangGraphAgent)
    agent.args = SimpleNamespace(model_api_key="secret")
    response_message = SimpleNamespace(
        content="I will validate the date boundary.",
        tool_calls=[],
        model_extra={},
    )
    agent._call_llm_with_retry = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=response_message)]
        )
    )
    state = {
        "messages": [],
        "item": {"instance_id": "task"},
        "conversation_history": [],
        "round_num": 2,
        "rollout_idx": 0,
        "performance": LangGraphAgent._new_performance(),
    }

    with caplog.at_level("INFO", logger="agent.langgraph_agent"):
        result = agent._call_model_node(state)

    assert "MODEL_ROUND instance=task rollout=1 round=3" in caplog.text
    assert "THINKING_OR_WORKING_NOTE:\nI will validate the date boundary." in caplog.text
    assert result["conversation_history"][-1]["reasoning_content"] == ""
