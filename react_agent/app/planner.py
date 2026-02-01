import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .llm import LLM
from .trace import Tracer
from .prompts import PLANNER_SYSTEM

@dataclass
class PlanStep:
    id: int
    goal: str
    tool_hint: str = ""

def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}

def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def plan(llm: LLM, tracer: Tracer, user_query: str) -> List[PlanStep]:
    resp = llm.respond(
        input_items=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_query},
        ],
        tools=None,
        tool_choice="none",
    )

    # 抽文本
    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        if _item_get(item, "type") in ("message", "output_text"):
            content = _item_get(item, "content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for c in content:
                    if _item_get(c, "type") in ("output_text", "text"):
                        chunks.append(_item_get(c, "text", ""))
    raw = "\n".join([c for c in chunks if c.strip()]).strip()
    tracer.log("planner.raw", raw=raw[:600])

    data = _safe_parse_json(raw)
    steps_data = data.get("steps", [])
    steps: List[PlanStep] = []
    if isinstance(steps_data, list):
        for s in steps_data:
            if not isinstance(s, dict):
                continue
            sid = s.get("id")
            goal = s.get("goal")
            tool_hint = s.get("tool_hint", "") or ""
            if isinstance(sid, int) and isinstance(goal, str) and goal.strip():
                steps.append(PlanStep(id=sid, goal=goal.strip(), tool_hint=str(tool_hint).strip()))
    if not steps:
        steps = [PlanStep(id=1, goal="Solve the user's request", tool_hint="")]

    tracer.log("planner.steps", steps=[{"id": x.id, "goal": x.goal, "tool_hint": x.tool_hint} for x in steps])
    return steps
