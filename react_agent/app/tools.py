import json
import math
from typing import Any, Callable, Dict, List

# -------------------------
# 1) 工具实现（Actions）
# -------------------------

def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return json.dumps({"ok": False, "error": "illegal characters"}, ensure_ascii=False)

    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

def lookup_doc(query: str) -> str:
    kb = {
        "react": "ReAct = Reasoning + Acting：模型在推理过程中按需调用工具，并用工具结果继续推理，直到得到答案。",
        "fsm": "FSM（有限状态机）= 状态集合 + 转移规则。Agent 可用 FSM 管理流程。",
        "responses api": "Responses API 统一输入/输出为 items，并原生支持 tool calling。",
    }
    q = query.lower().strip()
    hits = [v for k, v in kb.items() if k in q]
    return json.dumps({"ok": True, "hits": hits or []}, ensure_ascii=False)

# -------------------------
# 2) 工具 schema（给模型看的）
# -------------------------

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Safely evaluate a math expression. Returns JSON with {ok,result|error}.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "lookup_doc",
        "description": "Lookup internal docs/KB. Returns JSON with {ok,hits}.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

# -------------------------
# 3) 注册表（路由到工具实现）
# -------------------------

ToolFn = Callable[..., str]

TOOL_REGISTRY: Dict[str, ToolFn] = {
    "calculator": calculator,
    "lookup_doc": lookup_doc,
}
