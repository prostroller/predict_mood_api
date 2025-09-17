#!/bin/bash
# start.sh - Startup script untuk Railway

echo "=== Railway Deployment Debug ==="
echo "Environment variables:"
env | grep -E "(PORT|RAILWAY)" || echo "No RAILWAY/PORT env vars found"

echo "Starting mood detection API..."

# Handle PORT environment variable
if [ -z "$PORT" ]; then
    echo "No PORT env var, using 8000"
    export PORT=8000
else
    echo "Using PORT: $PORT"
fi

# Start the application
exec python mood_api_dummy.py