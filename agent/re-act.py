import os
import json
import math
import sys
from typing import Any, Dict, List

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ----------------------------
# 1) 定义你的“工具”(Actions)
# ----------------------------

def tool_calculator(expression: str) -> str:
    """
    一个很安全的示例计算器：只允许数字/运算符/括号/空格/小数点
    真实项目中建议用更严格的 parser 或者专门库。
    """
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return "ERROR: expression contains illegal characters"

    try:
        # 限制内建环境，避免 eval 注入
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def tool_lookup_doc(query: str) -> str:
    """
    模拟一个本地知识库/RAG 检索工具（这里用硬编码示例）。
    真实项目里你可以换成：向量库检索、SQL、ES、你自己的服务等。
    """
    kb = {
        "react": "ReAct = Reasoning + Acting：模型在思考过程中按需调用工具，并用工具结果继续推理，直到给出最终答案。",
        "responses api": "Responses API 是 OpenAI 新一代统一接口：输入由 items 组成，输出也由 items 组成，并天然支持 tool calling。",
        "fsm": "FSM（有限状态机）= 一组状态 + 转移规则。Agent 可以用 FSM 管理对话/任务流程。",
    }
    q = query.lower().strip()
    hits = [v for k, v in kb.items() if k in q]
    if not hits:
        return "No relevant doc found."
    return "\n".join(hits)

# 把工具实现注册到一个 dict，方便路由调用
TOOL_IMPL = {
    "calculator": tool_calculator,
    "lookup_doc": tool_lookup_doc,
}

# ----------------------------
# 2) 给模型声明“可调用工具”(Tool Schemas)
# ----------------------------
TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Evaluate a math expression and return the numeric result as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '12.5 * (3 + 4)'."}
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "lookup_doc",
        "description": "Lookup internal docs/knowledge base for a query and return relevant text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."}
            },
            "required": ["query"],
            "additionalProperties": False,
},
    },
]

# ----------------------------
# 3) ReAct Agent：循环直到模型给出 final text
# ----------------------------

SYSTEM_INSTRUCTIONS = """
You are a ReAct-style agent.

Rules:
- Use tools when they help. If you need calculation, call calculator.
- If you need internal knowledge, call lookup_doc.
- After tool results, incorporate them and continue.
- When you are ready, provide a concise final answer to the user.
"""

def run_react_agent(
    user_query: str,
    max_steps: int = 10,
    stream_output: bool = False,
    return_final_text: bool = True,
) -> str:
    def _get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        if hasattr(item, key):
            return getattr(item, key, default)
        if hasattr(item, "model_dump"):
            try:
                return item.model_dump().get(key, default)
            except Exception:
                return default
        return default

    def _as_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if hasattr(item, "model_dump"):
            try:
                return item.model_dump()
            except Exception:
                pass
        # Fallback: keep minimal shape so we can preserve type/id/name/arguments
        return {
            "type": _get(item, "type"),
            "id": _get(item, "id"),
            "call_id": _get(item, "call_id"),
            "name": _get(item, "name"),
            "arguments": _get(item, "arguments"),
            "content": _get(item, "content"),
        }

    def _get_text_parts(content: Any) -> List[str]:
        parts: List[str] = []
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            for c in content:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict):
                    if c.get("type") in ("output_text", "text"):
                        parts.append(c.get("text", ""))
                else:
                    c_type = _get(c, "type")
                    if c_type in ("output_text", "text"):
                        parts.append(_get(c, "text", ""))
        return parts

    # 用 items 记录对话与工具观测（Observation）
    # Responses API 的 input 支持一组 items（消息/工具结果等）
    conversation_items: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_query},
    ]

    for step in range(max_steps):
        if stream_output:
            stream = client.responses.create(
                model="gpt-4.1-mini",
                input=conversation_items,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            streamed_any = False
            final_response = None
            for event in stream:
                event_type = _get(event, "type")
                if event_type == "response.output_text.delta":
                    delta = _get(event, "delta", "")
                    if delta:
                        streamed_any = True
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                elif event_type in ("response.completed", "response.done"):
                    final_response = _get(event, "response")

            if streamed_any:
                sys.stdout.write("\n")
                sys.stdout.flush()

            if final_response is None:
                return "Streaming finished without a final response."

            output_items = _get(final_response, "output", [])
        else:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=conversation_items,
                tools=TOOLS,
                # 可选：让模型更“愿意”调用工具
                tool_choice="auto",
            )

            # Responses API：输出是 items。我们需要处理：
            # 1) function_call（模型要调用工具）
            # 2) message/text（模型最终回答或中间解释）
            # 不同 SDK 版本字段会略有差异，这里按“通用写法”做健壮处理。

            output_items = resp.output  # list of output items

        tool_called = False
        final_text_chunks: List[str] = []

        for item in output_items:
            item_type = _get(item, "type")

            # A) 模型请求调用工具
            if item_type in ("function_call", "tool_call"):
                tool_called = True
                tool_name = _get(item, "name")
                arguments = _get(item, "arguments") or "{}"
                call_id = _get(item, "call_id") or _get(item, "id")
                if not tool_name:
                    fn = _get(item, "function", {})
                    if isinstance(fn, dict):
                        tool_name = fn.get("name")
                        arguments = arguments or fn.get("arguments") or "{}"

                if tool_name:
                    sys.stdout.write(f"\n[tool] calling {tool_name} ...\n")
                    sys.stdout.flush()

                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    args = {}

                # 路由到本地工具实现
                if tool_name not in TOOL_IMPL:
                    tool_result = f"ERROR: unknown tool '{tool_name}'"
                else:
                    try:
                        tool_result = TOOL_IMPL[tool_name](**args)
                    except TypeError as e:
                        tool_result = f"ERROR: bad args for tool '{tool_name}': {e}"
                    except Exception as e:
                        tool_result = f"ERROR: tool '{tool_name}' failed: {type(e).__name__}: {e}"

                if tool_name:
                    sys.stdout.write(f"[tool] {tool_name} done.\n")
                    sys.stdout.flush()

                # 先把模型的 tool_call 放回对话历史，再追加 tool 输出
                conversation_items.append(_as_dict(item))
                # 把 Observation 回填给模型（关键：ReAct 的 “Observation”）
                conversation_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_result,
                })

            # B) 模型输出文本（可能是最终答案）
            elif item_type in ("message", "output_text"):
                # message 结构通常有 role/content
                content = _get(item, "content")
                final_text_chunks.extend(_get_text_parts(content))

        # 如果这一步没调用工具，并且产出了文本，我们认为它“收敛了”
        if (not tool_called) and final_text_chunks:
            final_text = "\n".join([t for t in final_text_chunks if t.strip()]).strip()
            if stream_output and not return_final_text:
                return ""
            return final_text

        # 如果没调用工具也没输出文本，避免死循环
        if not tool_called and not final_text_chunks:
            break

    return "Reached max steps without a final answer."

if __name__ == "__main__":
    q = "用一句话解释什么是ReAct，然后帮我算 (12.5 * (3 + 4)) / 5"
    result = run_react_agent(q, stream_output=True, return_final_text=False)
    if result:
        print(result)
