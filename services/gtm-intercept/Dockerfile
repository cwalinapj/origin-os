FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY gtm_intercept.py .
RUN useradd -m gtm
USER gtm
EXPOSE 8000
CMD ["uvicorn", "gtm_intercept:app", "--host", "0.0.0.0", "--port", "8000"]
