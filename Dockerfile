FROM python:3.11-slim

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY reengage_env/ ./reengage_env/
COPY server.py ./server.py
COPY openenv.yaml ./openenv.yaml
COPY scripts/ ./scripts/
COPY inference.py ./inference.py

# Healthcheck (must hit FastAPI)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Use non-root
USER appuser

EXPOSE 7860

ENV PORT=7860
ENV ENV_SEED=42
ENV PYTHONUNBUFFERED=1

# 🚨 CRITICAL: Run API, not UI
CMD ["python", "server.py"]
