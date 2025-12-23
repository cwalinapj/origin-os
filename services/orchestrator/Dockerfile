FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    websockets

COPY orchestrator.py .

RUN mkdir -p /data/orchestrator

ENV ORCHESTRATOR_DIR=/data/orchestrator

EXPOSE 8000

CMD ["python", "orchestrator.py"]
