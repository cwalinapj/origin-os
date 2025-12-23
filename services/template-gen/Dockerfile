FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic
COPY template_gen.py .
RUN useradd -m template
USER template
EXPOSE 8000
CMD ["uvicorn", "template_gen:app", "--host", "0.0.0.0", "--port", "8000"]
