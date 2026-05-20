FROM apache/airflow:2.10.4

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie dans /tmp pour éviter les problèmes de permissions
COPY shared/ /tmp/shared/
RUN chown -R airflow:root /tmp/shared

USER airflow

COPY airflow/requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt \
    && pip install --no-cache-dir /tmp/shared