FROM python:3.12-slim

# Install uv — the project's package manager.
RUN pip install --no-cache-dir uv==0.5.4

WORKDIR /app

# Copy lockfile + pyproject first for cache-friendly layer builds.
COPY pyproject.toml uv.lock* /app/

# Install runtime + api extras into /app/.venv
RUN uv sync --frozen --extra api || uv sync --extra api

# Copy the actual source.
COPY shared /app/shared
COPY projects /app/projects
COPY data /app/data

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "projects.rag_qa_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
