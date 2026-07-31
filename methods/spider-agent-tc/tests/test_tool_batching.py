import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

from agent.langgraph_agent import LangGraphAgent
from agent.message_processor import MessageProcessor
from servers import serve


def test_tool_batch_runs_concurrently_and_preserves_order(monkeypatch):
    calls = [
        {"name": "tool", "arguments": {"label": "first", "delay": 0.03}},
        {"name": "tool", "arguments": {"label": "second", "delay": 0.01}},
        {"name": "tool", "arguments": {"label": "third", "delay": 0}},
    ]
    active = 0
    peak_active = 0

    async def fake_execute_tool(name, **arguments):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(arguments["delay"])
        active -= 1
        return {"content": arguments["label"]}

    monkeypatch.setattr(serve.tool_registry, "has_tool", lambda name: True)
    monkeypatch.setattr(serve.tool_registry, "execute_tool", fake_execute_tool)

    results = asyncio.run(serve.execute_tool_batch(calls))

    assert peak_active == 3
    assert [result["content"] for result in results] == [
        "first",
        "second",
        "third",
    ]


def test_tool_batch_isolates_failures_and_unknown_tools(monkeypatch):
    calls = [
        {"name": "good", "arguments": {"value": 1}},
        {"name": "broken", "arguments": {}},
        {"name": "missing", "arguments": {}},
    ]

    async def fake_execute_tool(name, **arguments):
        if name == "broken":
            raise RuntimeError("private failure detail")
        return {"content": str(arguments["value"])}

    monkeypatch.setattr(
        serve.tool_registry,
        "has_tool",
        lambda name: name in {"good", "broken"},
    )
    monkeypatch.setattr(serve.tool_registry, "execute_tool", fake_execute_tool)

    results = asyncio.run(serve.execute_tool_batch(calls))

    assert results[0] == {"content": "1"}
    assert results[1] == {"error": "Tool execution failed: RuntimeError"}
    assert results[2] == {"error": "Tool missing not found"}
    assert "private failure detail" not in str(results)


def test_tool_batch_returns_safe_validation_feedback(monkeypatch):
    async def fake_execute_tool(name, **arguments):
        raise ValueError("Table is outside current database scope")

    monkeypatch.setattr(serve.tool_registry, "has_tool", lambda name: True)
    monkeypatch.setattr(serve.tool_registry, "execute_tool", fake_execute_tool)

    results = asyncio.run(
        serve.execute_tool_batch([{"name": "execute_sql", "arguments": {}}])
    )

    assert results == [
        {
            "error": "Table is outside current database scope",
            "error_type": "ValueError",
        }
    ]


def test_tool_server_profiles_contextualized_calls(monkeypatch):
    async def fake_execute_tool(name, **arguments):
        return {"content": '{"status": "success"}'}

    monkeypatch.setattr(serve.tool_registry, "has_tool", lambda name: True)
    monkeypatch.setattr(serve.tool_registry, "execute_tool", fake_execute_tool)

    result = asyncio.run(
        serve.execute_single_tool(
            {
                "name": "execute_sql",
                "arguments": {
                    "sql": "SELECT 1",
                    "_context": {"instance_id": "task"},
                },
            }
        )
    )

    assert result["content"] == '{"status": "success"}'
    assert result["_profile"]["duration_seconds"] >= 0


def test_message_processor_accepts_matching_result_list(monkeypatch):
    response = Mock()
    response.json.return_value = [{"content": "one"}, {"content": "two"}]
    monkeypatch.setattr("agent.message_processor.requests.post", Mock(return_value=response))
    processor = MessageProcessor(
        SimpleNamespace(api_host="127.0.0.1", api_port=5000, tool_request_timeout=1)
    )

    results = processor.execute_tool_calls(
        [
            {"name": "first", "arguments": {}},
            {"name": "second", "arguments": {}},
        ]
    )

    assert results == [{"content": "one"}, {"content": "two"}]
    response.raise_for_status.assert_called_once_with()


def test_message_processor_rejects_mismatched_result_count(monkeypatch):
    response = Mock()
    response.json.return_value = [{"content": "only one"}]
    monkeypatch.setattr("agent.message_processor.requests.post", Mock(return_value=response))
    processor = MessageProcessor(
        SimpleNamespace(api_host="127.0.0.1", api_port=5000, tool_request_timeout=1)
    )

    results = processor.execute_tool_calls(
        [
            {"name": "first", "arguments": {}},
            {"name": "second", "arguments": {}},
        ]
    )

    assert results == [
        {"error": "Tool API error: ValueError"},
        {"error": "Tool API error: ValueError"},
    ]


def test_agent_maps_ordered_results_to_every_tool_call_id(tmp_path):
    agent = object.__new__(LangGraphAgent)
    agent.message_processor = Mock()
    agent.message_processor.execute_tool_calls.return_value = [
        {"content": "first result"},
        {"content": "second result"},
    ]
    assistant_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-first",
                "name": "execute_sql",
                "args": {"sql": "SELECT 1"},
            },
            {
                "id": "call-second",
                "name": "search_schema",
                "args": {"query": "value"},
            },
        ],
    )
    state = {
        "messages": [assistant_message],
        "item": {
            "instance_id": "task",
            "db_id": "database",
            "instruction": "question",
        },
        "conversation_history": [],
        "round_num": 1,
        "rollout_idx": 0,
        "terminated": False,
        "error": None,
    }

    result = agent._execute_tools_node(state)

    assert [message.tool_call_id for message in result["messages"]] == [
        "call-first",
        "call-second",
    ]
    assert [message.content for message in result["messages"]] == [
        "first result",
        "second result",
    ]
    calls = agent.message_processor.execute_tool_calls.call_args.args[0]
    assert calls[0]["arguments"]["_context"]["allowed_database"] == "database"
    assert calls[1]["arguments"]["_context"]["instance_id"] == "task"


def test_agent_only_terminates_when_validator_accepts():
    agent = object.__new__(LangGraphAgent)
    agent.message_processor = Mock()
    assistant_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-terminate",
                "name": "terminate",
                "args": {"answer": "SELECT 1"},
            }
        ],
    )
    state = {
        "messages": [assistant_message],
        "item": {
            "instance_id": "task",
            "db_id": "database",
            "instruction": "question",
        },
        "conversation_history": [],
        "round_num": 1,
        "rollout_idx": 0,
        "terminated": False,
        "error": None,
    }

    agent.message_processor.execute_tool_calls.return_value = [
        {"content": '{"accepted": false, "reason": "not executed"}'}
    ]
    rejected = agent._execute_tools_node(state)
    assert rejected.get("terminated") is not True
    assert rejected["messages"][0].tool_call_id == "call-terminate"

    agent.message_processor.execute_tool_calls.return_value = [
        {"content": '{"accepted": true, "sql_sha256": "abc"}'}
    ]
    accepted = agent._execute_tools_node(state)
    assert accepted["terminated"] is True
    assert accepted["messages"] == []


def test_agent_collects_sql_and_termination_performance():
    agent = object.__new__(LangGraphAgent)
    agent.message_processor = Mock()
    assistant_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-query",
                "name": "execute_sql",
                "args": {"sql": "SELECT 123"},
            }
        ],
    )
    state = {
        "messages": [assistant_message],
        "item": {
            "instance_id": "task",
            "db_id": "database",
            "instruction": "question",
        },
        "conversation_history": [],
        "round_num": 1,
        "rollout_idx": 0,
        "terminated": False,
        "error": None,
        "performance": agent._new_performance(),
    }
    agent.message_processor.execute_tool_calls.return_value = [
        {
            "content": '{"status": "error", "sql_chars": 250}',
            "_profile": {"duration_seconds": 1.25},
        }
    ]

    result = agent._execute_tools_node(state)
    profile = result["performance"]

    assert profile["tool_calls"] == 1
    assert profile["tool_errors"] == 1
    assert profile["sql_calls"] == 1
    assert profile["sql_errors"] == 1
    assert profile["sql_duration_seconds"] == 1.25
    assert profile["max_sql_chars"] == 250


def test_performance_summary_reports_slowest_and_p95():
    profiles = {}
    for index in range(1, 21):
        profile = LangGraphAgent._new_performance()
        profile.update(
            {
                "duration_seconds": float(index),
                "model_calls": 1,
                "model_attempts": 1,
                "sql_calls": 2,
                "sql_duration_seconds": 0.5,
                "max_sql_chars": index * 10,
            }
        )
        profiles[f"task-{index:02d}"] = profile

    summary = LangGraphAgent._aggregate_performance(profiles)

    assert summary["profiled_tasks"] == 20
    assert summary["average_task_duration_seconds"] == 10.5
    assert summary["p95_task_duration_seconds"] == 19.0
    assert summary["slowest_task"] == {
        "instance_id": "task-20",
        "duration_seconds": 20.0,
    }
    assert summary["model_calls"] == 20
    assert summary["sql_calls"] == 40
    assert summary["max_sql_chars"] == 200


def test_agent_rejects_terminate_batched_with_other_tools():
    agent = object.__new__(LangGraphAgent)
    agent.message_processor = Mock()
    assistant_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-query",
                "name": "execute_sql",
                "args": {"sql": "SELECT 1"},
            },
            {
                "id": "call-terminate",
                "name": "terminate",
                "args": {"answer": "SELECT 1"},
            },
        ],
    )
    state = {
        "messages": [assistant_message],
        "item": {
            "instance_id": "task",
            "db_id": "database",
            "instruction": "question",
        },
        "conversation_history": [],
        "round_num": 1,
        "rollout_idx": 0,
        "terminated": False,
        "error": None,
    }

    result = agent._execute_tools_node(state)

    assert result.get("terminated") is not True
    assert len(result["messages"]) == 2
    assert all(
        "terminate must be the only tool call" in message.content
        for message in result["messages"]
    )
    agent.message_processor.execute_tool_calls.assert_not_called()
