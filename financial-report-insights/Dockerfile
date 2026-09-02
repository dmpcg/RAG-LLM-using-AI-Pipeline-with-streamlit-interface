# ---- Stage 1: Build dependencies ----
# WS-1 P1-F3 (2026-05-07): base image pinned to digest for reproducibility.
# Refresh via Renovate or `docker buildx imagetools inspect python:3.12-slim`.
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
# Same digest pin (WS-1 P1-F3).
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

# Prevent .pyc files and enable real-time log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install tini for proper signal handling (graceful SIGTERM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (top-level modules + subdirectory packages)
COPY *.py ./
COPY observability/ ./observability/

# Create directories with correct permissions
RUN mkdir -p /app/documents /app/.cache /app/.streamlit && \
    chown -R appuser:appuser /app

# Create minimal Streamlit config
RUN printf '[server]\nheadless = true\nmaxUploadSize = 200\n' > /app/.streamlit/config.toml && \
    chown appuser:appuser /app/.streamlit/config.toml

# Switch to non-root user
USER appuser

# Expose Streamlit + API ports
EXPOSE 8501 8504

# Health check verifies BOTH Streamlit and FastAPI are responding.
# Each urllib call uses an explicit timeout so a hung service cannot
# consume the full --timeout budget and mask a failed second service.
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
try: urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5) \
except Exception as e: print('Streamlit unhealthy:', e); sys.exit(1); \
try: urllib.request.urlopen('http://localhost:8504/health', timeout=5) \
except Exception as e: print('API unhealthy:', e); sys.exit(1)" || exit 1

# Use tini as init process for proper signal forwarding
ENTRYPOINT ["/usr/bin/tini", "--"]

# Process supervisor: exits container if EITHER process dies, triggering restart policy
CMD ["bash", "-c", "\
    uvicorn api:app --host 0.0.0.0 --port ${RAG_API_PORT:-8504} --log-level info & \
    UVICORN_PID=$!; \
    streamlit run streamlit_app_local.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --browser.gatherUsageStats=false & \
    STREAMLIT_PID=$!; \
    trap 'kill $UVICORN_PID $STREAMLIT_PID 2>/dev/null; exit 1' TERM INT; \
    while kill -0 $UVICORN_PID 2>/dev/null && kill -0 $STREAMLIT_PID 2>/dev/null; do \
        sleep 5; \
    done; \
    echo 'Process exited unexpectedly, shutting down...'; \
    kill $UVICORN_PID $STREAMLIT_PID 2>/dev/null; \
    exit 1"]
