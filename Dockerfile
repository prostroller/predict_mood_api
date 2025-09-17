# Multi-stage build for ultra-small image
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage - minimal base
FROM python:3.10-slim

WORKDIR /app

# Copy only the installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY mood_api_minimal.py .
COPY model/ model/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Start the application with minimal API
CMD ["python", "-m", "uvicorn", "mood_api_minimal:app", "--host", "0.0.0.0", "--port", "8000"]