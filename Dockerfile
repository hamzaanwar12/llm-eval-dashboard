FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src src
RUN uv sync --frozen --no-dev && useradd -r -u 10001 app && mkdir /data && chown app /data
USER app
ENV LLMOBS_DB=/data/llmobs.db
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "llmobs.app:app", "--host", "0.0.0.0", "--port", "8000"]
