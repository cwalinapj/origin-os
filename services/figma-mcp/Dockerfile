FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

COPY figma_mcp.py .

RUN useradd -m figma
USER figma

EXPOSE 8000

CMD ["uvicorn", "figma_mcp:app", "--host", "0.0.0.0", "--port", "8000"]
