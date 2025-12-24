FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    numpy \
    fastapi \
    uvicorn \
    redis \
    motor \
    prometheus-client \
    pydantic

# Copy LAM Forge code
COPY . .

# Create model directory
RUN mkdir -p /models/lam

EXPOSE 8060 8120 8130

CMD ["python", "-m", "uvicorn", "inference_spawner:app", "--host", "0.0.0.0", "--port", "8060"]
