"""Internal Agent entrypoint used by the YAML launcher."""

from .langgraph_agent import LangGraphAgent


def run_agent(runtime_args, items):
    """Run the agent with an already validated runtime configuration."""
    return LangGraphAgent(runtime_args).run(items)
