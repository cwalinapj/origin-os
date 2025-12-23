FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY commerce_mcp.py .
RUN useradd -m commerce
USER commerce
EXPOSE 8000
CMD ["uvicorn", "commerce_mcp:app", "--host", "0.0.0.0", "--port", "8000"]
