import os
from typing import Any, Dict, List, Optional
from openai import OpenAI

class LLM:
    """
    对 Responses API 的轻封装：你以后要换模型、加 retries、加超时、加日志都放这里。
    """
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def respond(
        self,
        input_items: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ):
        return self.client.responses.create(
            model=self.model,
            input=input_items,
            tools=tools,
            tool_choice=tool_choice,
        )
