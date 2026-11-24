FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
RUN uv sync --no-dev --no-editable

FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --create-home appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY data ./data

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV INDEX_PATH="/app/data/index.bin"
ENV SEMANTIC_PATH="/app/data/index.npy"

RUN findex index /app/data -o /app/data/index.bin --positions \
    && findex embed /app/data/index.bin -o /app/data/index.npy \
    && chown -R appuser:appgroup /app

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "findex serve --host ${HOST} --port ${PORT} --workers ${WORKERS:-1}"]
