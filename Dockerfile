# ReEngageEnv — Dockerfile
# Compatible with Hugging Face Spaces (openenv tag)
# Build:  docker build -t reengage-env .
# Run:    docker run -p 7860:7860 reengage-env

FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY reengage_env/ ./reengage_env/
COPY server.py     ./server.py
COPY openenv.yaml  ./openenv.yaml

# Optionally copy scripts for baseline runs
COPY scripts/      ./scripts/

# HF Space healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Non-root for HF Spaces compliance
USER appuser

EXPOSE 7860

ENV PORT=7860
ENV ENV_SEED=42
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
