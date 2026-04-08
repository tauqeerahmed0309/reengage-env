FROM python:3.11-slim

# HF Spaces non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY reengage_env/ ./reengage_env/
COPY server.py ./server.py
COPY app.py ./app.py          # <- important
COPY openenv.yaml ./openenv.yaml
COPY scripts/ ./scripts/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

USER appuser
EXPOSE 7860
ENV PORT=7860
ENV ENV_SEED=42
ENV PYTHONUNBUFFERED=1

# Start the UI app
CMD ["python", "app.py"]
