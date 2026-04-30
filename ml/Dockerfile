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

# Sécurité : Création de l'utilisateur
RUN useradd -m -u 1000 appuser

# Installation des outils globaux (en root)
RUN pip install --no-cache-dir --upgrade pip jupyterlab ipykernel

COPY --chown=appuser:appuser shared/ /app/shared/
COPY --chown=appuser:appuser ml/requirements.txt /app/ml-requirements.txt

#USER root
#RUN mkdir -p /mlflow && chmod 777 /mlflow
#RUN mkdir -p /app/mlflow && chmod 777 /app/mlflow

# Installation des dépendances (en utilisateur appuser)
USER appuser
RUN pip install --user --no-cache-dir -r /app/ml-requirements.txt \
    && pip install --user /app/shared

# Configuration Jupyter (nécessite root pour le sys-prefix)
USER root
RUN python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (Vélib)"

# Copie du code source
COPY --chown=appuser:appuser ml/ /app/ml/

USER appuser

# Port pour Jupyter (8888)
EXPOSE 8888

# Lancement du script principal
CMD ["python", "-u", "ml/src/main.py"]