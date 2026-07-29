"""Backward-compatible alias for the LangGraph-based agent.

The legacy hand-rolled loop and <tool_call> text parsing were replaced by
LangGraphAgent (agent/langgraph_agent.py), which uses OpenAI official tool
calling. This module preserves the original import paths:

- ``from agent.llm_agent import LLMAgent`` keeps working.
- ``LLMAgent.call_llm`` keeps its legacy signature for existing tests and
  callers, delegating to the same retrying OpenAI client call.
"""

from __future__ import annotations

from .langgraph_agent import LangGraphAgent


class LLMAgent(LangGraphAgent):
    """Drop-in replacement for the legacy agent, powered by LangGraph."""

    def call_llm(self, messages, instance_id=None, round_num=None):
        """Legacy helper: call the model and return assistant text content.

        Retained for backward compatibility with tests and external callers.
        Returns the assistant message content string, or an "ERROR: ..." string
        after exhausting retries (matching the legacy behavior).
        """
        try:
            response = self._call_llm_with_retry(
                messages, instance_id=instance_id, round_num=round_num
            )
        except Exception:
            return (
                f"ERROR: Failed to get response after "
                f"{self.args.retry['max_attempts']} attempts"
            )

        content = response.choices[0].message.content
        if content:
            return content
        return "ERROR: Failed to get response after retries"
