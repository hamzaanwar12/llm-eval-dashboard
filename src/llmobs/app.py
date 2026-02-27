"""Read-only dashboard API + page."""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from .store import Store

store = Store(os.getenv("LLMOBS_DB", "llmobs.db"))
app = FastAPI(title="llmobs")


@app.get("/api/traces")
def traces():
    return store.trace_stats()


@app.get("/api/evals")
def evals():
    return store.eval_runs()


@app.get("/api/evals/{run}")
def eval_cases(run: str):
    return store.eval_cases(run)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
