import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .llm import LLM
from .trace import Tracer
from .prompts import ROUTER_SYSTEM

AVAILABLE_TOOL_NAMES = ["calculator", "lookup_doc"]

@dataclass
class RouteDecision:
    route: str               # "direct" | "react"
    tools: List[str]         # recommended tools
    reason: str

def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}

def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def route(llm: LLM, tracer: Tracer, user_query: str) -> RouteDecision:
    # Router 不需要 tools，只要产出结构化 JSON 决策
    resp = llm.respond(
        input_items=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user_query},
        ],
        tools=None,
        tool_choice="none",
    )

    # 从 resp.output 抽文本
    text_chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        if _item_get(item, "type") in ("message", "output_text"):
            content = _item_get(item, "content")
            if isinstance(content, str):
                text_chunks.append(content)
            elif isinstance(content, list):
                for c in content:
                    if _item_get(c, "type") in ("output_text", "text"):
                        text_chunks.append(_item_get(c, "text", ""))
    raw = "\n".join([t for t in text_chunks if t.strip()]).strip()

    tracer.log("router.raw", raw=raw[:400])

    data = _safe_parse_json(raw)
    route_val = data.get("route", "react")
    tools = data.get("tools", [])
    reason = data.get("reason", "")

    # 规范化
    if route_val not in ("direct", "react"):
        route_val = "react"
    if not isinstance(tools, list):
        tools = []
    tools = [t for t in tools if t in AVAILABLE_TOOL_NAMES]

    if not reason or not isinstance(reason, str):
        reason = "no reason provided"

    decision = RouteDecision(route=route_val, tools=tools, reason=reason)
    tracer.log("router.decision", route=decision.route, tools=decision.tools, reason=decision.reason)
    return decision
