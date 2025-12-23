FROM python:3.11-slim
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn httpx

# Copy MCP code
COPY mcp.py .
COPY jwt_auth.py .

EXPOSE 8000
CMD ["python", "mcp.py"]
