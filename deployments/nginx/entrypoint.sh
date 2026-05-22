#!/bin/sh
set -e

CERT_DIR=/etc/nginx/certs

if [ ! -f "$CERT_DIR/server.crt" ] || [ ! -f "$CERT_DIR/server.key" ]; then
    echo "[nginx] Certificat SSL absent — génération automatique..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/server.key" \
        -out    "$CERT_DIR/server.crt" \
        -subj   "/C=FR/ST=Ile-de-France/L=Paris/O=MLOps-Velib/CN=localhost" \
        2>/dev/null
    echo "[nginx] Certificat généré dans $CERT_DIR"
fi

exec nginx -g "daemon off;"
