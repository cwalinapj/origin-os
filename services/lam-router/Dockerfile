FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic numpy
COPY lam_router.py .
RUN useradd -m lam
USER lam
EXPOSE 8000
CMD ["uvicorn", "lam_router:app", "--host", "0.0.0.0", "--port", "8000"]
