import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLM
from .memory import Memory
from .trace import Tracer
from .tools import TOOLS_SCHEMA, TOOL_REGISTRY
from .router import route as route_decide, RouteDecision
from .planner import plan as make_plan, PlanStep
from .prompts import EXECUTOR_SYSTEM

class State(str, Enum):
    ROUTE = "ROUTE"
    PLAN = "PLAN"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    EXECUTE = "EXECUTE"
    FINAL = "FINAL"
    STOP = "STOP"

@dataclass
class AgentConfig:
    max_tool_steps: int = 10
    enable_planner: bool = True

class AgentFSM:
    def __init__(self, llm: LLM, tracer: Tracer, config: AgentConfig = AgentConfig()):
        self.llm = llm
        self.tracer = tracer
        self.config = config

        self.state: State = State.ROUTE
        self.user_query: str = ""
        self.decision: Optional[RouteDecision] = None
        self.plan_steps: List[PlanStep] = []
        self.memory = Memory()
        self.final_answer: Optional[str] = None

    # --------- public ----------
    def run(self, user_query: str) -> str:
        self._reset(user_query)

        while self.state not in (State.FINAL, State.STOP):
            self.tracer.log("fsm.state", state=self.state)

            if self.state == State.ROUTE:
                self._state_route()
            elif self.state == State.PLAN:
                self._state_plan()
            elif self.state == State.DIRECT_ANSWER:
                self._state_direct_answer()
            elif self.state == State.EXECUTE:
                self._state_execute()
            else:
                self.state = State.STOP

        return self.final_answer or "Stopped without a final answer."

    # --------- states ----------
    def _state_route(self) -> None:
        self.decision = route_decide(self.llm, self.tracer, self.user_query)

        if self.decision.route == "direct":
            self.state = State.DIRECT_ANSWER
            return

        # react
        if self.config.enable_planner:
            self.state = State.PLAN
        else:
            self.state = State.EXECUTE

    def _state_plan(self) -> None:
        self.plan_steps = make_plan(self.llm, self.tracer, self.user_query)
        self.state = State.EXECUTE

    def _state_direct_answer(self) -> None:
        # 直接用 LLM 输出，不允许工具
        resp = self.llm.respond(
            input_items=[
                {"role": "system", "content": "You answer directly. Be concise and correct."},
                {"role": "user", "content": self.user_query},
            ],
            tools=None,
            tool_choice="none",
        )
        text = self._extract_text(resp)
        self.final_answer = text or "No answer."
        self.state = State.FINAL

    def _state_execute(self) -> None:
        """
        Executor：严格 ReAct 循环
        - 把 planner steps 作为“执行提示”写进 memory（可选）
        - 让模型 tool_call -> 我们执行 -> 回填 observation -> 继续
        """
        # 初始化 executor memory
        self.memory.add({"role": "system", "content": EXECUTOR_SYSTEM})

        if self.plan_steps:
            plan_text = "Plan:\n" + "\n".join(
                [f"{s.id}. {s.goal} (tool_hint={s.tool_hint or 'none'})" for s in self.plan_steps]
            )
            self.memory.add({"role": "developer", "content": plan_text})

        # 用户问题
        self.memory.add({"role": "user", "content": self.user_query})

        for step in range(self.config.max_tool_steps):
            items = self.memory.snapshot()
            self.tracer.log("executor.llm.request", step=step, items_len=len(items))

            resp = self.llm.respond(
                input_items=items,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )

            tool_calls = self._extract_tool_calls(resp)
            if tool_calls:
                self.tracer.log("executor.tool_calls", step=step, count=len(tool_calls))
                output_items = [self._normalize_item(i) for i in getattr(resp, "output", []) or []]
                if output_items:
                    self.memory.extend(output_items)
                for call in tool_calls:
                    obs = self._run_one_tool(self._normalize_item(call))
                    self.memory.add(obs)
                continue

            text = self._extract_text(resp)
            self.tracer.log("executor.text", step=step, text=text[:200])
            if text:
                self.final_answer = text
                self.state = State.FINAL
                return

            self.tracer.log("executor.stop", reason="empty_response")
            break

        self.final_answer = "Reached max tool steps without a final answer."
        self.state = State.FINAL

    # --------- helpers ----------
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

    def _reset(self, user_query: str) -> None:
        self.state = State.ROUTE
        self.user_query = user_query
        self.decision = None
        self.plan_steps = []
        self.memory = Memory()
        self.final_answer = None
        self.tracer.log("fsm.reset", query=user_query[:200])

    def _extract_tool_calls(self, resp) -> List[Dict[str, Any]]:
        calls = []
        for item in getattr(resp, "output", []) or []:
            if self._item_get(item, "type") in ("function_call", "tool_call"):
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
                        if self._item_get(c, "type") in ("output_text", "text"):
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

        output_type = "function_call_output" if call_type == "function_call" else "tool_output"
        return {"type": output_type, "call_id": call_id, "output": output}
