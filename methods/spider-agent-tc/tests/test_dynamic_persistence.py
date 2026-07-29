import json
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

from agent.file_manager import FileManager
from agent.langgraph_agent import LangGraphAgent


def test_upsert_rollout_result_replaces_progress_record(tmp_path):
    manager = FileManager(SimpleNamespace(output_folder=str(tmp_path)))
    manager.upsert_rollout_result(
        {
            "instance_id": "task",
            "rollout_idx": 0,
            "conversation": [{"role": "assistant", "content": "first"}],
            "terminated": False,
            "in_progress": True,
        }
    )
    manager.upsert_rollout_result(
        {
            "instance_id": "task",
            "rollout_idx": 0,
            "conversation": [{"role": "assistant", "content": "second"}],
            "terminated": True,
            "in_progress": False,
        }
    )

    saved = json.loads((tmp_path / "task.json").read_text(encoding="utf-8"))

    assert len(saved) == 1
    assert saved[0]["conversation"][0]["content"] == "second"
    assert saved[0]["terminated"] is True
    assert manager.processed_instances["task"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_process_single_item_refreshes_file_after_each_graph_step(tmp_path):
    item = {"instance_id": "task", "db_id": "database"}
    args = SimpleNamespace(
        output_folder=str(tmp_path),
        rollout_number=1,
        model_api_key="secret",
    )
    agent = object.__new__(LangGraphAgent)
    agent.args = args
    agent.processed_instances = {"task": 0}
    agent.file_manager = FileManager(args)
    agent.prompt_builder = SimpleNamespace(
        build_initial_prompt=lambda current_item, current_args: [
            {"role": "user", "content": "question"}
        ]
    )

    observations = []

    class StreamingGraph:
        def stream(self, initial_state, stream_mode):
            assert stream_mode == "values"
            assistant = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "execute_snowflake_sql",
                        "args": {"sql": "SELECT 1"},
                    }
                ],
            )
            model_state = {
                **initial_state,
                "messages": [*initial_state["messages"], assistant],
                "conversation_history": [
                    *initial_state["conversation_history"],
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "execute_snowflake_sql",
                                "arguments": {"sql": "SELECT 1"},
                            }
                        ],
                    },
                ],
                "round_num": 1,
            }
            yield model_state

            observations.append(
                json.loads(
                    (tmp_path / "task.json").read_text(encoding="utf-8")
                )[0]
            )
            tool_message = ToolMessage(content="1", tool_call_id="call-1")
            tool_state = {
                **model_state,
                "messages": [*model_state["messages"], tool_message],
                "conversation_history": [
                    *model_state["conversation_history"],
                    {"role": "tool", "content": "1"},
                ],
            }
            yield tool_state

            observations.append(
                json.loads(
                    (tmp_path / "task.json").read_text(encoding="utf-8")
                )[0]
            )

    agent.graph = StreamingGraph()

    result = agent.process_single_item(item, 0)
    saved = json.loads((tmp_path / "task.json").read_text(encoding="utf-8"))

    assert [len(record["conversation"]) for record in observations] == [2, 3]
    assert all(record["in_progress"] is True for record in observations)
    assert len(saved) == 1
    assert saved[0]["in_progress"] is False
    assert saved[0]["round_num"] == 1
    assert result == saved[0]


def test_stream_failure_keeps_last_persisted_conversation(tmp_path):
    item = {"instance_id": "task", "db_id": "database"}
    args = SimpleNamespace(
        output_folder=str(tmp_path),
        rollout_number=1,
        model_api_key="secret",
    )
    agent = object.__new__(LangGraphAgent)
    agent.args = args
    agent.processed_instances = {"task": 0}
    agent.file_manager = FileManager(args)
    agent.prompt_builder = SimpleNamespace(
        build_initial_prompt=lambda current_item, current_args: [
            {"role": "user", "content": "question"}
        ]
    )

    class FailingGraph:
        def stream(self, initial_state, stream_mode):
            yield {
                **initial_state,
                "messages": [
                    *initial_state["messages"],
                    AIMessage(content="working"),
                ],
                "conversation_history": [
                    *initial_state["conversation_history"],
                    {
                        "role": "assistant",
                        "content": "working",
                        "tool_calls": [],
                    },
                ],
                "round_num": 1,
            }
            raise RuntimeError("stream failed")

    agent.graph = FailingGraph()

    result = agent.process_single_item(item, 0)
    saved = json.loads((tmp_path / "task.json").read_text(encoding="utf-8"))[0]

    assert saved["conversation"][-1]["content"] == "working"
    assert saved["error"] == "stream failed"
    assert saved["in_progress"] is False
    assert result == saved
