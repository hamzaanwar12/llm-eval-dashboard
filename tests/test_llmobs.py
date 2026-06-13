from types import SimpleNamespace as NS
import pytest
from fastapi.testclient import TestClient
from llmobs import judge as judge_mod
from llmobs.store import Store, pct
from llmobs.trace import cost, traced


def msg(i=100, o=50, model="claude-opus-5-5"):
    return NS(model=model, usage=NS(input_tokens=i, output_tokens=o))


def test_pct():
    vals = list(range(1, 101))
    assert pct(vals, 50) == 50 and pct(vals, 95) == 95 and pct([], 50) == 0.0 and pct([7], 95) == 7


def test_cost():
    assert cost("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    assert cost("unknown", 10, 10) == 0.0


def test_traced_records_success_and_error():
    s = Store()

    @traced(s, "chat")
    def ok():
        return msg()

    @traced(s, "chat")
    def boom():
        raise TimeoutError

    ok(); ok()
    with pytest.raises(TimeoutError):
        boom()
    [st] = s.trace_stats()
    assert st["calls"] == 3 and st["error_rate"] == round(1 / 3, 4) and st["tokens"] == 300
    assert st["cost_usd"] == round(2 * cost("claude-opus-5-5", 100, 50), 6)


def fake_judge_client(score_for):
    def parse(**kw):
        import json
        data = json.loads(kw["messages"][0]["content"])
        return NS(stop_reason="end_turn", parsed_output=judge_mod.Verdict(score=score_for(data), reason="ok"))
    return NS(beta=NS(messages=NS(parse=parse)))


def test_run_eval_scores_and_survives_errors():
    s = Store()
    cases = [{"id": 1, "input": "2+2", "expected": "4"}, {"id": 2, "input": "capital of France", "expected": "Paris"},
             {"id": 3, "input": "crash", "expected": "x"}]

    def sut(q):
        if q == "crash":
            raise RuntimeError
        return {"2+2": "4", "capital of France": "Lyon"}[q]

    client = fake_judge_client(lambda d: 5 if d["candidate_answer"] == d["reference_answer"] else 2)
    run = judge_mod.run_eval(s, cases, sut, client=client, run="r1")
    assert s.eval_runs() == [{"run": "r1", "cases": 3, "avg_score": 2.67, "pass_rate": round(1 / 3, 4)}]
    worst = s.eval_cases(run)[0]
    assert worst["score"] == 1 and worst["reason"] == "error: RuntimeError"


def test_judge_refusal_is_scored_not_crashing():
    client = NS(beta=NS(messages=NS(parse=lambda **kw: NS(stop_reason="refusal", parsed_output=None))))
    assert judge_mod.judge(client, "q", "a", "e").score == 1


def test_api(monkeypatch):
    from llmobs import app as appmod
    s = Store()
    s.add("traces", {"name": "x", "ts": 0, "latency_ms": 12.0, "model": "m", "input_tokens": 1,
                     "output_tokens": 1, "cost_usd": 0, "error": None})
    monkeypatch.setattr(appmod, "store", s)
    c = TestClient(appmod.app)
    assert c.get("/api/traces").json()[0]["p50_ms"] == 12.0
    assert c.get("/api/evals").json() == []
    assert c.get("/").status_code == 200
