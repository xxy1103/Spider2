import argparse
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import logging

from config import load_config
from safe_logging import RedactingFilter, configured_sensitive_values
from servers.utils.tool_registry import ToolRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Tools Server API")
tool_registry = ToolRegistry()

@app.post("/execute")
async def execute_tool(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        tool_calls = data.get("tool_calls", [])
        
        if not tool_calls:
            return JSONResponse(
                status_code=400,
                content={"error": "No tool_calls provided"}
            )
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        logger.info(f"Executing tool: {tool_name}")
        
        if not tool_registry.has_tool(tool_name):
            return JSONResponse(
                status_code=404,
                content={"error": f"Tool {tool_name} not found"}
            )
        
        result = await tool_registry.execute_tool(tool_name, **arguments)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing request: {type(e).__name__}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {type(e).__name__}"}
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
