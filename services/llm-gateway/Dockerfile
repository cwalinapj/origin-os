FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

COPY gateway.py .

EXPOSE 8200

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8200"]
