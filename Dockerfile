FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000 \
    WEB_CONCURRENCY=1 \
    GUNICORN_THREADS=1 \
    GUNICORN_TIMEOUT=180 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    EPIGRANET_TORCH_THREADS=1 \
    EPIGRANET_TORCH_INTEROP_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py pipeline.py gunicorn.conf.py ./
COPY templates ./templates
COPY static ./static
COPY models ./models
COPY "class_mapping_209 (1).json" ./

EXPOSE 10000

CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]
