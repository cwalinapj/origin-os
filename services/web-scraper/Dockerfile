FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic
COPY web_scraper.py .
RUN useradd -m scraper
USER scraper
EXPOSE 8000
CMD ["uvicorn", "web_scraper:app", "--host", "0.0.0.0", "--port", "8000"]
