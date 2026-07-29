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
    agent.args = SimpleNamespace(databases_path=str(tmp_path))
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
                "name": "execute_snowflake_sql",
                "args": {"sql": "SELECT 1"},
            },
            {
                "id": "call-second",
                "name": "execute_bash",
                "args": {"command": "pwd"},
            },
        ],
    )
    state = {
        "messages": [assistant_message],
        "item": {"instance_id": "task", "db_id": "database"},
        "conversation_history": [],
        "round_num": 1,
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
