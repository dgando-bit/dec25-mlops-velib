FROM apache/airflow:2.11.2-python3.12

USER root

# Docker CLI + Compose plugin — requis pour que le scheduler lance
# 'docker compose run' via Docker-out-of-Docker (socket monté en volume)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && ARCH=$(dpkg --print-architecture) \
    && CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian ${CODENAME} stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       docker-ce-cli \
       docker-compose-plugin \
       jq \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-http
