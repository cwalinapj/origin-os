FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    numpy \
    b2sdk

COPY memory_service.py .

# Create data directories
RUN mkdir -p /data/memory/conversations \
             /data/memory/pending_sync \
             /data/memory/compacted \
             /data/memory/embeddings \
             /data/memory/index

# Environment defaults
ENV EBS_DATA_DIR=/data/memory
ENV EBS_MAX_SIZE_GB=100
ENV EBS_COMPACT_TRIGGER_GB=85
ENV EBS_TARGET_SIZE_GB=70
ENV SYNC_INTERVAL_SECONDS=60
ENV COMPACTION_CHECK_INTERVAL_SECONDS=300
ENV COMPACTION_MIN_AGE_HOURS=24
ENV COMPACTION_MIN_TURNS=10

EXPOSE 8000

CMD ["python", "memory_service.py"]
