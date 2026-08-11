# FastAPI and TensorFlow inference image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    MODEL_PATH=/app/models/densenet121_advanced_best.keras \
    PREDICTION_THRESHOLD=0.5

WORKDIR /app

# TensorFlow uses libgomp for CPU parallelism.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

# Copy only the files needed for inference, not training data or notebooks.
COPY --chown=10001:10001 api ./api
COPY --chown=10001:10001 \
    src/__init__.py \
    src/preprocess.py \
    src/gradcam.py \
    src/predict.py \
    ./src/
COPY --chown=10001:10001 \
    models/densenet121_advanced_best.keras \
    ./models/densenet121_advanced_best.keras

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
