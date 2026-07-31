import asyncio
from typing import Dict, Any, Callable, List, Optional
import logging
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)

class ToolRegistry:

    def __init__(self):
        self.tools = {}
        self.executor = None
        self.workers_per_tool = 8
    
    def set_workers_per_tool(self, workers: int):
 
        self.workers_per_tool = workers

        if self.executor:
            self.executor.shutdown(wait=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_per_tool)
    
    def register_tool(self, name: str, func: Callable):
        if name in self.tools:
            logger.warning(f"Tool {name} already registered, overwriting")
        self.tools[name] = func
        logger.info(f"Registered tool: {name}")
    
    def has_tool(self, name: str) -> bool:
        return name in self.tools
    
    def load_tools(self, config=None):
        logger.info("Loading tools...")

        from servers import structured_tools

        structured_tools.configure(config)
        structured_tools.register_tools(self)
        
        logger.info(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

        if not self.executor:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_per_tool)
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not registered")
        
        tool_func = self.tools[name]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            partial(tool_func, **kwargs)
        )
        
        return result
