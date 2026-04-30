FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances système si nécessaire (ex: curl pour le healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API et du dossier partagé
COPY api/ ./api/
COPY shared/ ./shared/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]