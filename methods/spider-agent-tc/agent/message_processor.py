import requests


class MessageProcessor:
    """Executes tool calls against the FastAPI tool server.

    The legacy text-parsing helpers (parse_assistant_message / parse_tool_calls /
    process_round) were removed when the agent loop migrated to LangGraph with
    official OpenAI tool calling. Only the HTTP execution path is preserved.
    """

    def __init__(self, args):
        self.args = args

    def execute_tool_calls(self, tool_calls):
        """Execute tool calls via API"""
        if not tool_calls:
            return []
            
        url = f"http://{self.args.api_host}:{self.args.api_port}/execute"
        request_body = {"tool_calls": tool_calls}
        
        try:
            response = requests.post(
                url, 
                json=request_body, 
                timeout=self.args.tool_request_timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            api_response = response.json()
            if not isinstance(api_response, list):
                raise ValueError("Tool server response must be a list")
            if len(api_response) != len(tool_calls):
                raise ValueError(
                    f"Expected {len(tool_calls)} tool results, "
                    f"received {len(api_response)}"
                )
            if not all(isinstance(result, dict) for result in api_response):
                raise ValueError("Every tool result must be an object")
            return api_response
                
        except Exception as e:
            error_result = {"error": f"Tool API error: {type(e).__name__}"}
            return [error_result] * len(tool_calls)
