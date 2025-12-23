FROM python:3.11-slim

WORKDIR /app

# Install dependencies including boto3 for S3
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    cryptography \
    pydantic \
    boto3

# Copy both versions - S3 is primary
COPY vault_service_s3.py vault_service.py ./

# Create cache directory
RUN mkdir -p /tmp/vault-cache

ENV CACHE_DIR=/tmp/vault-cache

EXPOSE 8000

# Use S3 version by default
CMD ["python", "vault_service_s3.py"]
