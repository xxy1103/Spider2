import argparse
import asyncio
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import load_config
from safe_logging import RedactingFilter, configured_sensitive_values
from servers.utils.tool_registry import ToolRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Tools Server API")
tool_registry = ToolRegistry()


async def execute_single_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool call without cancelling sibling calls on failure."""
    if not isinstance(tool_call, dict):
        return {"error": "Invalid tool call: expected an object"}

    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if not isinstance(tool_name, str) or not tool_name:
        return {"error": "Invalid tool call: missing tool name"}
    if not isinstance(arguments, dict):
        return {"error": f"Invalid arguments for tool {tool_name}: expected an object"}

    logger.info("Executing tool: %s", tool_name)
    if not tool_registry.has_tool(tool_name):
        return {"error": f"Tool {tool_name} not found"}

    try:
        return await tool_registry.execute_tool(tool_name, **arguments)
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool execution failed for %s: %s", tool_name, type(exc).__name__)
        return {"error": f"Tool execution failed: {type(exc).__name__}"}


async def execute_tool_batch(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Execute a batch concurrently while preserving input order."""
    return await asyncio.gather(
        *(execute_single_tool(tool_call) for tool_call in tool_calls)
    )


@app.post("/execute")
async def execute_tools(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        tool_calls = data.get("tool_calls", [])
        
        if not isinstance(tool_calls, list) or not tool_calls:
            return JSONResponse(
                status_code=400,
                content={"error": "tool_calls must be a non-empty list"}
            )
        
        results = await execute_tool_batch(tool_calls)
        return JSONResponse(content=results)
    
    except Exception as exc:  # noqa: BLE001
        logger.error("Error processing request: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {type(exc).__name__}"}
        )

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "tools": sorted(tool_registry.tools)}

def parse_args():
    parser = argparse.ArgumentParser(description="Tools Server")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    parser.add_argument("--port", required=True, type=int, help="Resolved server port")
    parser.add_argument("--host", required=True, type=str, help="Server host")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    redacting_filter = RedactingFilter(configured_sensitive_values(config))
    for handler in logging.getLogger().handlers:
        handler.addFilter(redacting_filter)
    tool_registry.load_tools(config)
    workers = config.raw["server"]["workers_per_tool"]
    tool_registry.set_workers_per_tool(workers)
    
    logger.info(f"Starting server on port {args.port} with {workers} workers per tool")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
