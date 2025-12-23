FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx
COPY ui.py .
EXPOSE 8000
CMD ["python", "ui.py"]
