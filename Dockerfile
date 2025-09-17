# Ultra-minimal Dockerfile - No ML dependencies - Force rebuild
FROM python:3.10.15-slim

WORKDIR /app

# Install only FastAPI and uvicorn (no gcc needed) 
RUN pip install --no-cache-dir fastapi==0.85.0 uvicorn==0.18.0 pydantic==1.10.2

# Copy only API code
COPY mood_api_dummy.py .

# Create user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Run Python script directly which handles PORT env var
CMD ["python", "mood_api_dummy.py"]