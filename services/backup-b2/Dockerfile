FROM python:3.11-slim

WORKDIR /app

# Install boto3 for S3 API and b2 CLI
RUN pip install --no-cache-dir boto3 b2

# Copy all Python scripts
COPY *.py .
COPY *.yaml .

CMD ["python", "backup_worm.py"]
