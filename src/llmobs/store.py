"""SQLite storage for traces and eval results, plus aggregate stats."""
import math
import sqlite3
import threading

SCHEMA = """
create table if not exists traces(id integer primary key, name text, ts real, latency_ms real, model text,
    input_tokens integer, output_tokens integer, cost_usd real, error text);
create table if not exists evals(id integer primary key, run text, ts real, case_id text, input text,
    output text, expected text, score integer, reason text);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()  # shortcut: single-process writer; use Postgres/ClickHouse for many services
        self.c.executescript(SCHEMA)

    def add(self, table: str, row: dict):
        assert table in ("traces", "evals")
        cols = ", ".join(row)
        with self.lock:
            self.c.execute(f"insert into {table}({cols}) values ({', '.join('?' * len(row))})", tuple(row.values()))
            self.c.commit()

    def trace_stats(self) -> list[dict]:
        rows = self.c.execute("select name, latency_ms, cost_usd, error, input_tokens + output_tokens from traces").fetchall()
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r[0], []).append(r)
        out = []
        for name, rs in sorted(by.items()):
            lat = sorted(r[1] for r in rs)
            out.append({"name": name, "calls": len(rs), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
                        "error_rate": round(sum(r[3] is not None for r in rs) / len(rs), 4),
                        "tokens": sum(r[4] or 0 for r in rs), "cost_usd": round(sum(r[2] or 0 for r in rs), 6)})
        return out

    def eval_runs(self) -> list[dict]:
        rows = self.c.execute("""select run, count(*), avg(score), sum(score >= 4), min(ts) from evals
                                 group by run order by min(ts) desc""").fetchall()
        return [{"run": r[0], "cases": r[1], "avg_score": round(r[2], 2), "pass_rate": round(r[3] / r[1], 4)} for r in rows]

    def eval_cases(self, run: str) -> list[dict]:
        cur = self.c.execute("select case_id, input, output, expected, score, reason from evals where run=? order by score",
                             (run,))
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


def pct(sorted_vals: list[float], p: int) -> float:
    """Nearest-rank percentile."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, math.ceil(p / 100 * len(sorted_vals)) - 1))
    return round(sorted_vals[k], 1)
