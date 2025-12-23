FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    boto3 \
    pyyaml \
    cryptography \
    requests

COPY sentinel.py .
COPY sentinel.schema.yaml .

# Run as non-root
RUN useradd -m sentinel
USER sentinel

CMD ["python", "sentinel.py"]
