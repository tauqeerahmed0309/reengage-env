FROM python:3.11-slim

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY reengage_env/ ./reengage_env/
COPY server.py ./server.py
COPY app.py ./app.py
COPY openenv.yaml ./openenv.yaml
COPY scripts/ ./scripts/

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Environment variables
ENV PORT=7860
ENV ENV_SEED=42
ENV PYTHONUNBUFFERED=1

# Start app
CMD ["python", "app.py"]