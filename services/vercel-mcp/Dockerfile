FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

COPY vercel_mcp.py .

RUN useradd -m vercel
USER vercel

EXPOSE 8000

CMD ["uvicorn", "vercel_mcp:app", "--host", "0.0.0.0", "--port", "8000"]
