FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY lam_experiment.py .
RUN useradd -m experiment
USER experiment
EXPOSE 8000
CMD ["uvicorn", "lam_experiment:app", "--host", "0.0.0.0", "--port", "8000"]
