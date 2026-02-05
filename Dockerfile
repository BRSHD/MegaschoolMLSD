FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.linux.gpu.txt /app/requirements.linux.gpu.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.linux.gpu.txt

COPY app /app/app
COPY static /app/static

ENV GPU_ONLY=1 \
    CV_USE_GPU=1 \
    LLAMA_REQUIRE_GPU=1 \
    LLAMA_GPU_LAYERS=-1 \
    LLAMA_BATCH=128 \
    LLAMA_CTX=1024 \
    LLAMA_MAX_TOKENS=200 \
    CV_OCR_SCALE=1.3 \
    CV_OCR_DET_LIMIT=640 \
    CV_OCR_USE_ANGLE=0 \
    CV_YOLO_IMG=960 \
    MAX_REQUEST_SECONDS=20 \
    LOG_LEVEL=INFO \
    LOG_JSON=1

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
