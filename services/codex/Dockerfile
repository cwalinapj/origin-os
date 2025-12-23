FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn google-api-python-client google-auth
COPY codex.py .
EXPOSE 8000
CMD ["python", "codex.py"]
