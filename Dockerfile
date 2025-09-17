# Ultra-lightweight Docker build
FROM python:3.10-slim as builder

WORKDIR /build

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first to avoid CUDA downloads
RUN pip install --user torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy and install other dependencies
COPY requirements.txt .
RUN pip install --user transformers==4.21.0 fastapi==0.85.0 uvicorn==0.18.0 pydantic==1.10.2

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY mood_api_minimal.py .
COPY model/ model/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Update PATH for user packages
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "mood_api_minimal:app", "--host", "0.0.0.0", "--port", "8000"]