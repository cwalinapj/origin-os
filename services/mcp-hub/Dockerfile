FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    docker \
    pydantic

# Copy files
COPY mcp_hub.py .
COPY jwt_auth.py .

# Create data directory
RUN mkdir -p /data/memory

EXPOSE 8000

CMD ["python", "mcp_hub.py"]
