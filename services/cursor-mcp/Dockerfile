FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    python-dotenv \
    websockets \
    aiofiles

COPY cursor_mcp.py .
COPY cursor_client.py .
COPY mcp_tools.py .

RUN useradd -m cursor
USER cursor

EXPOSE 8000

CMD ["uvicorn", "cursor_mcp:app", "--host", "0.0.0.0", "--port", "8000"]
