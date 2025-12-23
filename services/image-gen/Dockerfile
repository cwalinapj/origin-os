FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY image_gen.py .
RUN useradd -m imagegen
USER imagegen
EXPOSE 8000
CMD ["uvicorn", "image_gen:app", "--host", "0.0.0.0", "--port", "8000"]
