FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY device_router.py .
RUN useradd -m router
USER router
EXPOSE 8000
CMD ["uvicorn", "device_router:app", "--host", "0.0.0.0", "--port", "8000"]
