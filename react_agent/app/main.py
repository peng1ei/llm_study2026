from .llm import LLM
from .trace import Tracer
from .agent_fsm import AgentFSM, AgentConfig

def main():
    tracer = Tracer()
    agent = AgentFSM(
        llm=LLM(model="gpt-4.1-mini"),
        tracer=tracer,
        config=AgentConfig(max_tool_steps=8, enable_planner=True),
    )

    q = "解释 ReAct 是什么，然后帮我算 (12.5*(3+4))/5，并给出结果。"
    ans = agent.run(q)

    print("\n===== ANSWER =====\n")
    print(ans)

    print("\n===== TRACE (tail 25) =====\n")
    tracer.print_tail(25)

if __name__ == "__main__":
    main()
