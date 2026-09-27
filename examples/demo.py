"""Trace a few real Claude calls and run a 3-case eval. Needs ANTHROPIC_API_KEY.

    uv run python examples/demo.py && uv run uvicorn llmobs.app:app
"""
import anthropic
from llmobs.judge import run_eval
from llmobs.store import Store
from llmobs.trace import traced

store = Store("llmobs.db")
client = anthropic.Anthropic()


@traced(store, "qa")
def ask(question: str):
    return client.messages.create(model="claude-opus-5-5", max_tokens=1000, output_config={"effort": "low"},
                                  messages=[{"role": "user", "content": f"Answer in one short sentence: {question}"}])


def answer(question: str) -> str:
    msg = ask(question)
    return "".join(b.text for b in msg.content if b.type == "text")


cases = [
    {"id": "geo-1", "input": "What is the capital of Australia?", "expected": "Canberra"},
    {"id": "math-1", "input": "What is 17 * 23?", "expected": "391"},
    {"id": "sci-1", "input": "What gas do plants absorb for photosynthesis?", "expected": "Carbon dioxide (CO2)"},
]
print("run:", run_eval(store, cases, answer, client=client))
print(store.trace_stats())
