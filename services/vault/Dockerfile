FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    cryptography \
    pydantic

COPY vault_service.py .

RUN mkdir -p /vault

ENV VAULT_DIR=/vault

EXPOSE 8000

CMD ["python", "vault_service.py"]
