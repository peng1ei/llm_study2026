from .agent import ReactAgent, AgentConfig
from .llm import LLM
from .memory import Memory
from .trace import Tracer

def main():
    tracer = Tracer()
    agent = ReactAgent(
        llm=LLM(model="gpt-4.1-mini"),
        memory=Memory(),
        tracer=tracer,
        config=AgentConfig(max_steps=8),
    )

    q = "先用一句话解释 ReAct，然后算 (12.5*(3+4))/5，并解释你为何需要工具。"
    ans = agent.run(q)
    print("\n===== ANSWER =====\n")
    print(ans)

    print("\n===== TRACE (last 20) =====\n")
    tracer.print_tail(20)

if __name__ == "__main__":
    main()
