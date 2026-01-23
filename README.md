# llm_study2026

## 文件目录
agent-from-scratch/
├── llm/
│   └── openai_client.py      # 模型调用（唯一依赖 OpenAI）
├── prompt/
│   └── system_prompt.py      # system prompt 管理
├── tools/
│   ├── base.py               # Tool 抽象
│   ├── calculator.py
│   └── search.py
├── agent/
│   ├── agent.py              # Agent 主循环（最核心）
│   └── state.py              # Agent State（messages / step）
├── memory/
│   ├── short_term.py
│   └── long_term.py
├── trace/
│   └── tracer.py             # 可观测性（强烈建议）
├── main.py                   # CLI / Demo
└── requirements.txt
tracer.py