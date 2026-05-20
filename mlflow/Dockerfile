FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY mlflow/requirements.txt /app/mlflow-requirements.txt
RUN pip install --no-cache-dir -r /app/mlflow-requirements.txt

RUN mkdir -p /app/mlflow/artifacts \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --serve-artifacts \
    --artifacts-destination /app/mlflow/artifacts