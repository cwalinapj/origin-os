FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

COPY registry_service.py .

RUN mkdir -p /data/registry

ENV REGISTRY_DIR=/data/registry

EXPOSE 8000

CMD ["python", "registry_service.py"]
