"""`@traced` decorator: records latency, tokens, cost and errors for any function returning an Anthropic Message."""
import functools
import time
from .store import Store

# USD per 1M tokens (input, output). Update when pricing changes.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1e6


def traced(store: Store, name: str | None = None):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t0, row = time.perf_counter(), {"name": name or fn.__name__, "ts": time.time(), "error": None}
            try:
                res = fn(*a, **kw)
                u = getattr(res, "usage", None)
                if u is not None:
                    model = getattr(res, "model", "")
                    row.update(model=model, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                               cost_usd=cost(model, u.input_tokens, u.output_tokens))
                return res
            except Exception as e:
                row["error"] = type(e).__name__
                raise
            finally:  # recorded on success and failure
                row["latency_ms"] = (time.perf_counter() - t0) * 1000
                store.add("traces", row)
        return wrapper
    return deco
