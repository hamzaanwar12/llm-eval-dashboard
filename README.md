# llmobs — LLM observability & evals

Know what your LLM features cost, how fast they are, and whether they're getting better.
A tiny, dependency-light toolkit: **trace calls**, **score outputs with an LLM judge**, **see it on a dashboard**.

## 1. Trace
```python
from llmobs.store import Store
from llmobs.trace import traced

store = Store("llmobs.db")

@traced(store, "support-bot")
def reply(q):
    return client.messages.create(model="claude-opus-5-5", max_tokens=1000, messages=[...])
```
Every call records latency, model, input/output tokens, **USD cost**, and the error type on failure.
The dashboard shows **p50/p95 latency, error rate, tokens and cost** per traced name.

## 2. Evaluate
```python
from llmobs.judge import run_eval
run_eval(store, [{"id": "1", "input": "17*23?", "expected": "391"}], my_system)
```
Claude grades each answer against the reference with **structured output** (`score` 1–5 + `reason`), using a
rubric that treats the candidate as data, not instructions. A crashing case scores 1 instead of killing the run;
judge refusals are handled. Runs are compared by average score and pass rate (score ≥ 4).

## 3. Dashboard
```bash
uv sync
export ANTHROPIC_API_KEY=...
uv run python examples/demo.py        # 3 traced calls + a 3-case eval
uv run uvicorn llmobs.app:app         # http://localhost:8000
```
Click a run to see its cases, worst first.

## Test
```bash
uv run pytest -q     # Claude is faked; no key needed
```

## Stack
Python · Anthropic SDK (structured outputs) · FastAPI · SQLite · vanilla JS · Docker · GitHub Actions
