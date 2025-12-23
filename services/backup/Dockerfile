FROM python:3.11-slim

WORKDIR /app

# Install Docker CLI for volume operations
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    schedule \
    boto3

COPY backup_service.py .

RUN mkdir -p /backups

ENV BACKUP_DIR=/backups
ENV BACKUP_RETENTION_DAYS=30
ENV BACKUP_SCHEDULE=daily

EXPOSE 8000

CMD ["python", "backup_service.py"]
