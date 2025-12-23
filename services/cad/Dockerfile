FROM python:3.11-slim

WORKDIR /app

# Install OpenSCAD and dependencies
RUN apt-get update && apt-get install -y \
    openscad \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    numpy-stl \
    trimesh

# Try to install CadQuery (may fail on some platforms)
RUN pip install --no-cache-dir cadquery || echo "CadQuery not available on this platform"

COPY cad_service.py .

RUN mkdir -p /data/stl

ENV CAD_OUTPUT_DIR=/data/stl
ENV DISPLAY=:99

EXPOSE 8000

CMD ["python", "cad_service.py"]
