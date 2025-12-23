FROM python:3.11-slim
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn google-api-python-client google-auth

# Copy Codex code
COPY codex.py .
COPY jwt_auth.py .

EXPOSE 8000
CMD ["python", "codex.py"]
