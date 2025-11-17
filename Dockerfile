# is-vector-search FastAPI service
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment managed by uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY app ./app

# Optionally copy runtime config (you may prefer to mount .env in production)
# COPY .env .

EXPOSE 8200

# Default command: run FastAPI app
CMD ["uv", "run", "python", "-m", "app.main"]
