import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLM
from .memory import Memory
from .tools import TOOLS_SCHEMA, TOOL_REGISTRY
from .trace import Tracer
from .prompts import SYSTEM_INSTRUCTIONS

@dataclass
class AgentConfig:
    max_steps: int = 10

class ReactAgent:
    def __init__(self, llm: LLM, memory: Memory, tracer: Tracer, config: AgentConfig = AgentConfig()):
        self.llm = llm
        self.memory = memory
        self.tracer = tracer
        self.config = config

    def _item_get(self, item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _normalize_item(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if hasattr(item, "dict"):
            return item.dict()
        return dict(item)

    def _extract_tool_calls(self, resp) -> List[Dict[str, Any]]:
        # 兼容处理：遍历 resp.output，找 function_call/tool_call
        calls = []
        for item in getattr(resp, "output", []) or []:
            t = self._item_get(item, "type")
            if t in ("function_call", "tool_call"):
                calls.append(item)
        return calls

    def _extract_text(self, resp) -> str:
        chunks: List[str] = []
        for item in getattr(resp, "output", []) or []:
            t = self._item_get(item, "type")
            if t in ("message", "output_text"):
                content = self._item_get(item, "content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for c in content:
                        c_type = self._item_get(c, "type")
                        if c_type in ("output_text", "text"):
                            chunks.append(self._item_get(c, "text", ""))
                        elif isinstance(c, str):
                            chunks.append(c)
        return "\n".join([c for c in chunks if c.strip()]).strip()

    def _run_one_tool(self, call_item: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = self._item_get(call_item, "name")
        arguments = self._item_get(call_item, "arguments") or "{}"
        call_id = self._item_get(call_item, "call_id") or self._item_get(call_item, "id")
        call_type = self._item_get(call_item, "type")

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        except json.JSONDecodeError:
            args = {}

        self.tracer.log("tool.call", name=tool_name, args=args, call_id=call_id)

        fn = TOOL_REGISTRY.get(tool_name)
        if not fn:
            output = json.dumps({"ok": False, "error": f"unknown tool: {tool_name}"}, ensure_ascii=False)
        else:
            try:
                output = fn(**args)
            except TypeError as e:
                output = json.dumps({"ok": False, "error": f"bad args: {e}"}, ensure_ascii=False)
            except Exception as e:
                output = json.dumps({"ok": False, "error": f"tool failed: {type(e).__name__}: {e}"}, ensure_ascii=False)

        self.tracer.log("tool.result", name=tool_name, call_id=call_id, output=output)

        # Observation item (function_call_output) 回填模型
        output_type = "function_call_output" if call_type == "function_call" else "tool_output"
        return {"type": output_type, "call_id": call_id, "output": output}

    def run(self, user_query: str) -> str:
        # 初始化上下文
        self.memory.add({"role": "system", "content": SYSTEM_INSTRUCTIONS})
        self.memory.add({"role": "user", "content": user_query})

        for step in range(self.config.max_steps):
            items = self.memory.snapshot()
            self.tracer.log("llm.request", step=step, items_len=len(items))

            resp = self.llm.respond(
                input_items=items,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )

            # 1) 先处理工具调用（Act）
            tool_calls = self._extract_tool_calls(resp)
            if tool_calls:
                self.tracer.log("llm.tool_calls", step=step, count=len(tool_calls))
                output_items = [self._normalize_item(i) for i in getattr(resp, "output", []) or []]
                if output_items:
                    self.memory.extend(output_items)
                for call in tool_calls:
                    obs = self._run_one_tool(self._normalize_item(call))
                    self.memory.add(obs)
                # 回到循环继续（Reason -> next Act/Final）
                continue

            # 2) 没有工具调用：尝试提取最终文本（Final）
            text = self._extract_text(resp)
            self.tracer.log("llm.text", step=step, text=text[:200])

            if text:
                return text

            # 3) 兜底：避免空响应死循环
            self.tracer.log("agent.stop", reason="empty_response")
            break

        return "Reached max steps without a final answer."
