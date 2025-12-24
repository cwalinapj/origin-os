FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic

COPY registry.py api.py ./

EXPOSE 8300

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8300"]
