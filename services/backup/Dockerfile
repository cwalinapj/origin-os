FROM python:3.11-slim

WORKDIR /app

# Install Docker CLI and rsync for backup operations
RUN apt-get update && apt-get install -y \
    docker.io \
    rsync \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Core dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    schedule

# Optional remote backup providers (install failures are OK)
RUN pip install --no-cache-dir b2sdk || echo "Backblaze B2 SDK not installed"
RUN pip install --no-cache-dir dropbox || echo "Dropbox SDK not installed"
RUN pip install --no-cache-dir google-api-python-client google-auth || echo "Google Drive SDK not installed"
RUN pip install --no-cache-dir boto3 || echo "AWS SDK not installed"

COPY backup_service.py .

RUN mkdir -p /backups /credentials

ENV BACKUP_DIR=/backups
ENV BACKUP_RETENTION_DAYS=30
ENV BACKUP_SCHEDULE=daily
ENV REMOTE_BACKUP_TYPE=local

EXPOSE 8000

CMD ["python", "backup_service.py"]
