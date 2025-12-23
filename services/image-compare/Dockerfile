FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic pillow numpy
COPY image_compare.py .
RUN useradd -m compare
USER compare
EXPOSE 8000
CMD ["uvicorn", "image_compare:app", "--host", "0.0.0.0", "--port", "8000"]
