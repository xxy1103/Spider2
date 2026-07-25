"""Internal Agent entrypoint used by the YAML launcher."""

from .llm_agent import LLMAgent


def run_agent(runtime_args, items):
    """Run the agent with an already validated runtime configuration."""
    return LLMAgent(runtime_args).run(items)
