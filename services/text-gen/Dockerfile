FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY text_gen.py .
RUN useradd -m textgen
USER textgen
EXPOSE 8000
CMD ["uvicorn", "text_gen:app", "--host", "0.0.0.0", "--port", "8000"]
