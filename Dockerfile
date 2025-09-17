# Ultra-minimal Dockerfile - Force complete rebuild v2
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy API file
COPY mood_api_dummy.py .

# Set environment and run
ENV PORT=8000
EXPOSE 8000

# Run directly with proper PORT handling
CMD python mood_api_dummy.py