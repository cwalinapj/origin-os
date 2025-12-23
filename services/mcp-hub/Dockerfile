FROM python:3.11-slim

WORKDIR /app

# Install git for git operations
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    docker \
    pydantic \
    boto3

# Copy files
COPY mcp_hub.py .
COPY jwt_auth.py .

# Create data directories
RUN mkdir -p /data/memory /data/repos

EXPOSE 8000

CMD ["python", "mcp_hub.py"]
