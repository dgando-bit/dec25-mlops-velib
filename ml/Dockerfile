FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

RUN pip install --no-cache-dir --upgrade pip \
    jupyterlab \
    ipykernel

# Copie des dépendances avant le code (cache Docker optimal)
COPY --chown=appuser:appuser shared/ /app/shared/
COPY --chown=appuser:appuser ml/requirements.txt /app/ml-requirements.txt

RUN pip install --no-cache-dir -r /app/ml-requirements.txt \
    && pip install --no-cache-dir /app/shared \
    && python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (Vélib)"

# Code source en dernier (invalidation de cache minimale)
COPY --chown=appuser:appuser ml/ /app/ml/

USER appuser

EXPOSE 8888

CMD ["python", "-u", "ml/src/main.py"]