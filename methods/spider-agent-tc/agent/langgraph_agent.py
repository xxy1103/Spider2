"""LangGraph-based agent implementation using OpenAI official tool calling.

This module replaces the legacy text-parsing agent loop with a LangGraph
StateGraph, while preserving the result format and execution lifecycle:
- Same initial prompt construction (via prompt_builders)
- Same tool execution path (HTTP POST to the FastAPI tool server)
- A terminate tool call succeeds only after server-side validation
- Same result persistence format (via FileManager)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI

from .file_manager import FileManager
from .message_processor import MessageProcessor
from .progress import TaskProgressReporter
from .prompt_builders import get_prompt_builder
from servers.structured_tools import get_openai_tools

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State carried through the LangGraph execution."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    item: dict[str, Any]
    conversation_history: list[dict[str, Any]]
    round_num: int
    rollout_idx: int
    terminated: bool
    error: str | None
    performance: dict[str, Any]


class LangGraphAgent:
    """Agent driven by a LangGraph StateGraph with official OpenAI tool calls."""

    def __init__(self, args):
        self.args = args
        self.model_client = OpenAI(
            base_url=args.model_base_url,
            api_key=args.model_api_key,
            timeout=args.model_request_timeout,
        )

        self.file_manager = FileManager(args)
        self.message_processor = MessageProcessor(args)
        self.prompt_builder = get_prompt_builder(args.prompt_strategy)
        self.tools = get_openai_tools()

        self.processed_instances = defaultdict(int)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # LLM invocation with retry (behavior preserved from legacy LLMAgent)
    # ------------------------------------------------------------------
    def _call_llm_with_retry(
        self,
        openai_messages,
        instance_id=None,
        round_num=None,
        attempt_callback=None,
    ):
        """Call LLM with retry mechanism. Returns the raw response or raises."""
        retry = self.args.retry
        max_attempts = retry["max_attempts"]
        delay = retry["initial_delay_seconds"]
        attempt = 0
        last_error: Exception | None = None

        while attempt < max_attempts:
            if attempt_callback is not None:
                attempt_callback()
            try:
                response = self.model_client.chat.completions.create(
                    model=self.args.model,
                    messages=openai_messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_new_tokens,
                    n=1,
                )
                return response
            except Exception as e:  # noqa: BLE001
                last_error = e
                attempt += 1
                instance_info = f" for {instance_id}" if instance_id else ""
                round_info = f" (round {round_num})" if round_num is not None else ""
                safe_message = str(e).replace(self.args.model_api_key, "***REDACTED***")
                logger.warning(
                    "LLM error%s%s: %s: %s. Attempt %s/%s failed.",
                    instance_info,
                    round_info,
                    type(e).__name__,
                    safe_message,
                    attempt,
                    max_attempts,
                )
                if attempt >= max_attempts:
                    raise
                time.sleep(delay)
                delay = min(
                    delay * retry["backoff_multiplier"],
                    retry["max_delay_seconds"],
                )
        raise last_error if last_error else RuntimeError("Unexpected exit from retry loop")

    # ------------------------------------------------------------------
    # Message format conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _to_openai_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to OpenAI wire format."""
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                wire: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    wire["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                openai_messages.append(wire)
            elif isinstance(msg, ToolMessage):
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                openai_messages.append({"role": "user", "content": str(msg.content)})
        return openai_messages

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _call_model_node(self, state: AgentState) -> dict[str, Any]:
        instance_id = state["item"]["instance_id"]
        round_num = state["round_num"] + 1
        openai_messages = self._to_openai_messages(state["messages"])
        performance = self._copy_performance(
            state.get("performance") or self._new_performance()
        )
        attempt_count = 0

        def count_attempt() -> None:
            nonlocal attempt_count
            attempt_count += 1

        started_at = time.perf_counter()
        try:
            response = self._call_llm_with_retry(
                openai_messages,
                instance_id=instance_id,
                round_num=round_num,
                attempt_callback=count_attempt,
            )
        except Exception as e:  # noqa: BLE001
            performance["model_calls"] += 1
            performance["model_attempts"] += attempt_count
            performance["model_errors"] += 1
            performance["model_duration_seconds"] += (
                time.perf_counter() - started_at
            )
            safe_message = str(e).replace(self.args.model_api_key, "***REDACTED***")
            return {
                "error": f"ERROR: Failed to get response after retries: {safe_message}",
                "performance": performance,
            }
        performance["model_calls"] += 1
        performance["model_attempts"] += attempt_count
        performance["model_duration_seconds"] += time.perf_counter() - started_at

        assistant_message = response.choices[0].message
        content = assistant_message.content or ""
        raw_tool_calls = assistant_message.tool_calls or []

        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": self._parse_tool_arguments(tc.function.arguments),
            }
            for tc in raw_tool_calls
        ]

        ai_message = AIMessage(content=content, tool_calls=tool_calls)

        conversation_history = state["conversation_history"] + [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"name": tc["name"], "arguments": tc["args"]} for tc in tool_calls
                ],
            }
        ]

        return {
            "messages": [ai_message],
            "conversation_history": conversation_history,
            "round_num": round_num,
            "performance": performance,
        }

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _execute_tools_node(self, state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        if len(last_message.tool_calls) > 1 and any(
            tc["name"] == "terminate" for tc in last_message.tool_calls
        ):
            reason = json.dumps(
                {
                    "accepted": False,
                    "reason": "terminate must be the only tool call in its assistant message.",
                }
            )
            tool_messages = [
                ToolMessage(content=reason, tool_call_id=tc["id"])
                for tc in last_message.tool_calls
            ]
            history_entries = [
                {"role": "tool", "content": reason}
                for _ in last_message.tool_calls
            ]
            performance = self._copy_performance(
                state.get("performance") or self._new_performance()
            )
            for tc in last_message.tool_calls:
                self._record_tool_profile(
                    performance,
                    tc["name"],
                    tc.get("args", {}),
                    duration_seconds=0.0,
                    failed=True,
                    terminate_rejected=tc["name"] == "terminate",
                )
            return {
                "messages": tool_messages,
                "conversation_history": (
                    state["conversation_history"] + history_entries
                ),
                "performance": performance,
            }

        tool_calls_for_execution: list[dict[str, Any]] = []
        for tc in last_message.tool_calls:
            arguments = dict(tc["args"])
            arguments["_context"] = {
                "instance_id": state["item"]["instance_id"],
                "rollout_idx": state["rollout_idx"],
                "allowed_database": state["item"]["db_id"],
                "instruction": state["item"]["instruction"],
                "external_knowledge": state["item"].get("external_knowledge"),
            }
            tool_calls_for_execution.append(
                {"name": tc["name"], "arguments": arguments}
            )

        exec_results = self.message_processor.execute_tool_calls(
            tool_calls_for_execution
        )

        tool_messages: list[ToolMessage] = []
        history_entries: list[dict[str, Any]] = []
        accepted_termination = False
        performance = self._copy_performance(
            state.get("performance") or self._new_performance()
        )
        for tc, exec_result in zip(last_message.tool_calls, exec_results):
            profile = exec_result.pop("_profile", {})
            result_content = exec_result.get("content", str(exec_result))
            failed = "error" in exec_result
            content_payload: dict[str, Any] = {}
            if isinstance(result_content, str):
                try:
                    parsed_content = json.loads(result_content)
                    if isinstance(parsed_content, dict):
                        content_payload = parsed_content
                except json.JSONDecodeError:
                    pass
            if content_payload.get("status") == "error":
                failed = True
            terminate_rejected = False
            if tc["name"] == "terminate":
                try:
                    accepted_termination = bool(
                        json.loads(result_content).get("accepted")
                    )
                except (json.JSONDecodeError, AttributeError):
                    accepted_termination = False
                terminate_rejected = not accepted_termination
            self._record_tool_profile(
                performance,
                tc["name"],
                tc.get("args", {}),
                duration_seconds=float(profile.get("duration_seconds", 0.0)),
                failed=failed,
                terminate_rejected=terminate_rejected,
                observed_sql_chars=content_payload.get("sql_chars"),
            )
            if tc["name"] == "terminate":
                if accepted_termination:
                    continue
            tool_messages.append(
                ToolMessage(content=result_content, tool_call_id=tc["id"])
            )
            history_entries.append({"role": "tool", "content": result_content})

        result: dict[str, Any] = {
            "messages": tool_messages,
            "conversation_history": state["conversation_history"] + history_entries,
            "performance": performance,
        }
        if accepted_termination:
            result["terminated"] = True
        return result

    @staticmethod
    def _new_performance() -> dict[str, Any]:
        return {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 0.0,
            "model_calls": 0,
            "model_attempts": 0,
            "model_errors": 0,
            "model_duration_seconds": 0.0,
            "tool_calls": 0,
            "tool_errors": 0,
            "tool_duration_seconds": 0.0,
            "sql_calls": 0,
            "sql_errors": 0,
            "sql_duration_seconds": 0.0,
            "max_sql_chars": 0,
            "terminate_calls": 0,
            "terminate_rejections": 0,
            "tools": {},
        }

    @staticmethod
    def _copy_performance(performance: dict[str, Any]) -> dict[str, Any]:
        copied = dict(performance)
        copied["tools"] = {
            name: dict(values)
            for name, values in performance.get("tools", {}).items()
        }
        return copied

    @staticmethod
    def _record_tool_profile(
        performance: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        *,
        duration_seconds: float,
        failed: bool,
        terminate_rejected: bool,
        observed_sql_chars: Any = None,
    ) -> None:
        performance["tool_calls"] += 1
        performance["tool_duration_seconds"] += duration_seconds
        if failed:
            performance["tool_errors"] += 1

        tool_stats = performance["tools"].setdefault(
            tool_name,
            {"calls": 0, "errors": 0, "duration_seconds": 0.0},
        )
        tool_stats["calls"] += 1
        tool_stats["duration_seconds"] += duration_seconds
        if failed:
            tool_stats["errors"] += 1

        if tool_name == "execute_sql":
            performance["sql_calls"] += 1
            performance["sql_duration_seconds"] += duration_seconds
            if failed:
                performance["sql_errors"] += 1
            sql = arguments.get("sql")
            if isinstance(sql, str):
                performance["max_sql_chars"] = max(
                    performance["max_sql_chars"], len(sql)
                )
            if isinstance(observed_sql_chars, int):
                performance["max_sql_chars"] = max(
                    performance["max_sql_chars"], observed_sql_chars
                )
        elif tool_name == "terminate":
            performance["terminate_calls"] += 1
            if terminate_rejected:
                performance["terminate_rejections"] += 1
            answer = arguments.get("answer")
            if isinstance(answer, str):
                performance["max_sql_chars"] = max(
                    performance["max_sql_chars"], len(answer)
                )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _route_after_model(self, state: AgentState) -> str:
        if state.get("error"):
            return "end"
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "execute_tools"
        if state["round_num"] >= self.args.max_rounds:
            return "end"
        return "call_model"

    def _route_after_tools(self, state: AgentState) -> str:
        if state.get("terminated"):
            return "end"
        if state["round_num"] >= self.args.max_rounds:
            return "end"
        return "call_model"

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("call_model", self._call_model_node)
        workflow.add_node("execute_tools", self._execute_tools_node)

        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {
                "execute_tools": "execute_tools",
                "call_model": "call_model",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "execute_tools",
            self._route_after_tools,
            {"call_model": "call_model", "end": END},
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    # Item processing (interface preserved from legacy LLMAgent)
    # ------------------------------------------------------------------
    def _result_from_state(
        self,
        state: AgentState,
        rollout_idx: int,
        *,
        in_progress: bool,
    ) -> dict[str, Any]:
        """Build the persisted rollout record for a graph state."""
        result = {
            "instance_id": state["item"]["instance_id"],
            "rollout_idx": rollout_idx,
            "conversation": state["conversation_history"],
            "final_messages": self._to_openai_messages(state["messages"]),
            "terminated": state.get("terminated", False),
            "in_progress": in_progress,
            "round_num": state.get("round_num", 0),
            "performance": self._copy_performance(
                state.get("performance") or self._new_performance()
            ),
        }
        if state.get("error"):
            result["error"] = state["error"]
            result["round_failed"] = state.get("round_num", 0)
        return result

    def process_single_item(self, item, rollout_idx):
        instance_id = item["instance_id"]
        last_state: AgentState | None = None
        task_started_at = time.perf_counter()

        if self.processed_instances[instance_id] >= self.args.rollout_number:
            logger.info(
                "Skipping %s rollout %s (already completed %s valid rollouts)",
                instance_id,
                rollout_idx + 1,
                self.processed_instances[instance_id],
            )
            return None

        try:
            initial_messages = self.prompt_builder.build_initial_prompt(item, self.args)
            lc_messages: list[BaseMessage] = [
                SystemMessage(content=m["content"])
                if m["role"] == "system"
                else HumanMessage(content=m["content"])
                for m in initial_messages
            ]

            initial_state: AgentState = {
                "messages": lc_messages,
                "item": item,
                "conversation_history": list(initial_messages),
                "round_num": 0,
                "rollout_idx": rollout_idx,
                "terminated": False,
                "error": None,
                "performance": self._new_performance(),
            }

            final_state = initial_state
            last_state = initial_state
            for current_state in self.graph.stream(
                initial_state,
                {"recursion_limit": self.args.max_rounds * 2 + 5},
                stream_mode="values",
            ):
                final_state = current_state
                last_state = current_state
                progress_result = self._result_from_state(
                    current_state,
                    rollout_idx,
                    in_progress=True,
                )
                self.file_manager.upsert_rollout_result(progress_result)

            result = self._result_from_state(
                final_state,
                rollout_idx,
                in_progress=False,
            )
            result["performance"]["duration_seconds"] = round(
                time.perf_counter() - task_started_at, 6
            )
            result["performance"]["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.file_manager.upsert_rollout_result(result)

            return result

        except Exception as e:  # noqa: BLE001
            safe_message = str(e).replace(self.args.model_api_key, "***REDACTED***")
            if last_state is not None:
                error_result = self._result_from_state(
                    last_state,
                    rollout_idx,
                    in_progress=False,
                )
                error_result["error"] = safe_message
                error_result["terminated"] = False
                error_result["round_failed"] = last_state.get("round_num", 0)
                error_result["performance"]["duration_seconds"] = round(
                    time.perf_counter() - task_started_at, 6
                )
                error_result["performance"]["finished_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
            else:
                performance = self._new_performance()
                performance["duration_seconds"] = round(
                    time.perf_counter() - task_started_at, 6
                )
                performance["finished_at"] = datetime.now(timezone.utc).isoformat()
                error_result = {
                    "instance_id": instance_id,
                    "rollout_idx": rollout_idx,
                    "error": safe_message,
                    "terminated": False,
                    "in_progress": False,
                    "performance": performance,
                }
            self.file_manager.upsert_rollout_result(error_result)
            logger.error(
                "Error processing %s rollout %s: %s",
                instance_id,
                rollout_idx + 1,
                safe_message,
            )
            return error_result

    def _run_progress_task(self, reporter, item, rollout_idx):
        reporter.task_started()
        try:
            result = self.process_single_item(item, rollout_idx)
            reporter.task_finished(
                success=bool(result and result.get("terminated") is True)
            )
            return result
        except Exception:
            reporter.task_finished(success=False)
            raise

    def run(self, items):
        """Main execution function (interface preserved)."""
        self.file_manager.load_existing_results()
        self.processed_instances = self.file_manager.processed_instances
        os.makedirs(self.args.output_folder, exist_ok=True)

        tasks_to_process = []
        for item in items:
            instance_id = item["instance_id"]
            current_valid_rollouts = self.processed_instances[instance_id]
            for rollout_idx in range(current_valid_rollouts, self.args.rollout_number):
                tasks_to_process.append((item, rollout_idx))

        if not tasks_to_process:
            with TaskProgressReporter("Agent", 0):
                pass
            return self.build_summary(items)

        with TaskProgressReporter("Agent", len(tasks_to_process)) as reporter:
            with ThreadPoolExecutor(max_workers=self.args.num_threads) as executor:
                futures = [
                    executor.submit(
                        self._run_progress_task, reporter, item, rollout_idx
                    )
                    for item, rollout_idx in tasks_to_process
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            "Unexpected worker failure: %s", type(exc).__name__
                        )

        return self.build_summary(items)

    def build_summary(self, items):
        """Summarize completion from persisted per-instance result files."""
        successful = []
        failed = []
        profiles: dict[str, dict[str, Any]] = {}
        required_rollouts = self.args.rollout_number
        for item in items:
            instance_id = item["instance_id"]
            results = self.file_manager.load_instance_results(instance_id)
            completed_results = [
                result
                for result in results
                if isinstance(result, dict) and not result.get("in_progress", False)
            ]
            if completed_results:
                profile = completed_results[0].get("performance")
                if isinstance(profile, dict):
                    profiles[instance_id] = profile
            terminated = sum(
                1
                for result in results
                if isinstance(result, dict) and result.get("terminated") is True
            )
            if terminated >= required_rollouts:
                successful.append(instance_id)
            else:
                failed.append(instance_id)
        return {
            "total_tasks": len(items),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "successful_instance_ids": successful,
            "failed_instance_ids": failed,
            "rollout_number": required_rollouts,
            "performance": self._aggregate_performance(profiles),
        }

    @staticmethod
    def _aggregate_performance(
        profiles: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not profiles:
            return {"profiled_tasks": 0, "tasks": {}}

        durations = sorted(
            float(profile.get("duration_seconds", 0.0))
            for profile in profiles.values()
        )
        p95_index = max(0, (len(durations) * 95 + 99) // 100 - 1)
        slowest_instance_id = max(
            profiles,
            key=lambda instance_id: float(
                profiles[instance_id].get("duration_seconds", 0.0)
            ),
        )
        tools: dict[str, dict[str, Any]] = {}
        for profile in profiles.values():
            for tool_name, values in profile.get("tools", {}).items():
                aggregate = tools.setdefault(
                    tool_name,
                    {"calls": 0, "errors": 0, "duration_seconds": 0.0},
                )
                aggregate["calls"] += int(values.get("calls", 0))
                aggregate["errors"] += int(values.get("errors", 0))
                aggregate["duration_seconds"] += float(
                    values.get("duration_seconds", 0.0)
                )

        summed_fields = [
            "model_calls",
            "model_attempts",
            "model_errors",
            "model_duration_seconds",
            "tool_calls",
            "tool_errors",
            "tool_duration_seconds",
            "sql_calls",
            "sql_errors",
            "sql_duration_seconds",
            "terminate_calls",
            "terminate_rejections",
        ]
        summary: dict[str, Any] = {
            "profiled_tasks": len(profiles),
            "average_task_duration_seconds": round(
                sum(durations) / len(durations), 6
            ),
            "p95_task_duration_seconds": round(durations[p95_index], 6),
            "slowest_task": {
                "instance_id": slowest_instance_id,
                "duration_seconds": round(
                    float(
                        profiles[slowest_instance_id].get(
                            "duration_seconds", 0.0
                        )
                    ),
                    6,
                ),
            },
            "max_sql_chars": max(
                int(profile.get("max_sql_chars", 0))
                for profile in profiles.values()
            ),
            "tools": tools,
            "tasks": profiles,
        }
        for field in summed_fields:
            summary[field] = round(
                sum(float(profile.get(field, 0)) for profile in profiles.values()),
                6,
            )
            if not field.endswith("_seconds"):
                summary[field] = int(summary[field])
        for values in summary["tools"].values():
            values["duration_seconds"] = round(values["duration_seconds"], 6)
        return summary
