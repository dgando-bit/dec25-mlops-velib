# Setup Nginx avec SSL — Guide étape par étape

## Contexte

Ce projet MLOps expose trois services internes :

| Service       | Container name                  | Port interne |
|---------------|---------------------------------|-------------|
| API FastAPI   | `velib_api`                     | 8000        |
| MLflow        | `dec25-mlops-mlflow-server`     | 5000        |
| Jupyter Lab   | `dec25-mlops-jupyter`           | 8888        |

L'objectif est de placer Nginx devant ces services comme **reverse proxy SSL-terminating** :
- Toutes les connexions HTTP (port 80) sont redirigées vers HTTPS (port 443).
- Nginx gère le certificat et transmet le trafic déchiffré aux services en interne.

> **Zero-friction** : les certificats SSL sont générés automatiquement par le container Nginx
> au premier démarrage via un script d'entrée. Aucune dépendance à installer sur la machine hôte.

---

## Pour démarrer (TL;DR)

```bash
git clone <repo>
cd dec25-mlops-velib
cp .env.example .env   # remplir les variables
make setup
make build
make up
```

C'est tout. Les URLs HTTPS sont disponibles dès que les containers sont démarrés.

---

## Architecture

```
Internet / Browser
        │
    port 80 (HTTP)
        │
   ┌────▼────┐
   │  Nginx  │──── redirect 301 → HTTPS
   │  :443   │
   └────┬────┘
        │ SSL termination (certificat auto-signé)
        │
   ┌────┴──────────────────────────────┐
   │         Routes HTTPS              │
   │  /         → api:8000             │
   │  /mlflow/  → mlflow-server:5000   │
   │  /jupyter/ → jupyter-service:8888 │
   └───────────────────────────────────┘
        réseau interne Docker: mlops-net
```

---

## Étape 1 — Script d'entrée (`entrypoint.sh`)

**Fichier :** `deployments/nginx/entrypoint.sh`

```sh
#!/bin/sh
set -e

CERT_DIR=/etc/nginx/certs

if [ ! -f "$CERT_DIR/server.crt" ] || [ ! -f "$CERT_DIR/server.key" ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$CERT_DIR/server.key" \
        -out    "$CERT_DIR/server.crt" \
        -subj   "/CN=localhost"
fi

exec nginx -g "daemon off;"
```

**Logique :** au démarrage du container, le script vérifie si les fichiers `server.crt` et `server.key`
existent dans le volume monté. S'ils sont absents (premier lancement), OpenSSL les génère.
Sinon, il les réutilise — le certificat n'est donc généré qu'une seule fois.

`exec nginx -g "daemon off;"` remplace le processus shell par Nginx, ce qui est la bonne pratique
dans les containers Docker (PID 1 = le processus principal).

---

## Étape 2 — Dockerfile Nginx

**Fichier :** `deployments/nginx/Dockerfile`

```dockerfile
FROM nginx:alpine

RUN apk add --no-cache openssl

COPY nginx.conf    /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/entrypoint.sh"]
```

- `nginx:alpine` : image officielle allégée (~10 Mo).
- `apk add openssl` : installe OpenSSL dans l'image pour la génération des certs.
- `chmod +x` : rend le script exécutable (nécessaire sur Linux).
- `ENTRYPOINT` : le script est le point d'entrée, et non la commande par défaut de Nginx.

---

## Étape 3 — Configuration Nginx (`nginx.conf`)

**Fichier :** `deployments/nginx/nginx.conf`

### Resolver DNS — résolution lazy à la requête

```nginx
resolver 127.0.0.11 valid=30s;
```

`127.0.0.11` est le serveur DNS interne de Docker. Avec cette directive, Nginx résout
les noms de services **à la requête** et non au démarrage. Cela évite le crash
`host not found in upstream` lorsqu'un backend est absent ou pas encore prêt.

Les adresses sont ensuite passées via des variables `set` dans chaque `location` :

```nginx
location / {
    set $api_upstream api:8000;
    proxy_pass http://$api_upstream;
}
```

> **Pourquoi ne pas utiliser des blocs `upstream` ?**
> Les blocs `upstream { server api:8000; }` sont résolus une fois au démarrage de Nginx.
> Si le container `api` n'existe pas encore à cet instant, Nginx refuse de démarrer.
> Les variables `set` + `resolver` déplacent la résolution DNS au moment de la requête.

### Redirection HTTP → HTTPS

```nginx
server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}
```

`server_name _` est un catch-all : accepte n'importe quel `Host`.

### Serveur HTTPS

```nginx
server {
    listen 443 ssl;
    ssl_certificate     /etc/nginx/certs/server.crt;
    ssl_certificate_key /etc/nginx/certs/server.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_session_cache   shared:SSL:10m;
    ...
}
```

- **TLSv1.2 TLSv1.3** : TLS 1.0 et 1.1 sont désactivés (obsolètes, vulnérables).
- **ssl_session_cache** : évite de renegocier le handshake TLS à chaque requête.

### Headers de sécurité

```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Content-Type-Options nosniff always;
add_header X-Frame-Options SAMEORIGIN always;
```

### Routing

| Chemin      | Destination            | Note                                        |
|-------------|------------------------|---------------------------------------------|
| `/`         | `api:8000`             | Route par défaut                            |
| `/mlflow/`  | `mlflow-server:5000`   | `rewrite` pour strip le préfixe             |
| `/jupyter/` | `jupyter-service:8888` | Préfixe conservé + WebSocket activé         |

**Strip du préfixe MLflow** — MLflow sert à la racine `/`, on retire donc `/mlflow/`
avant de transmettre. Avec des variables, on ne peut pas utiliser le trailing slash de
`proxy_pass`, on utilise `rewrite` à la place :
```nginx
location /mlflow/ {
    set $mlflow_upstream mlflow-server:5000;
    rewrite ^/mlflow/(.*) /$1 break;
    proxy_pass http://$mlflow_upstream;
}
```

**Jupyter — préfixe conservé** — Jupyter est configuré avec `base_url=/jupyter/`, il
attend donc des requêtes qui commencent par `/jupyter/`. On ne retire pas le préfixe :
```nginx
location /jupyter/ {
    set $jupyter_upstream jupyter-service:8888;
    proxy_pass http://$jupyter_upstream;  # URI transmise telle quelle
}
```

**WebSocket pour Jupyter** — sans ces lignes, les kernels ne répondent pas :
```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade    $http_upgrade;
proxy_set_header Connection "upgrade";
```

---

## Étape 4 — Service dans `docker-compose.yml`

```yaml
nginx:
  container_name: dec25-mlops-nginx
  build:
    context: ./deployments/nginx
    dockerfile: Dockerfile
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./deployments/nginx/certs:/etc/nginx/certs  # rw : entrypoint écrit les certs
  depends_on:
    - api
    - mlflow-server
    - jupyter-service
  networks:
    - mlops-net
  restart: unless-stopped
```

Le volume bind-mount `./deployments/nginx/certs` permet aux certificats générés à l'intérieur
du container d'être persistés sur le disque hôte. Ils survivent aux redémarrages.

### Mise à jour Jupyter (`base_url`)

```yaml
command: >
  jupyter lab ... --ServerApp.base_url=/jupyter/
```

Sans cette option, les assets JS/CSS de Jupyter seraient chargés depuis `/static/...`
au lieu de `/jupyter/static/...`, cassant l'interface.

---

## Étape 5 — `.gitignore`

```gitignore
deployments/nginx/certs/*.key
deployments/nginx/certs/*.crt
```

La clé privée ne doit **jamais** être versionnée. Le dossier `certs/` est présent dans le repo
via un `.gitkeep`, mais son contenu (les fichiers générés) est ignoré.

---

## Étape 6 — Makefile

La cible `make setup` crée le dossier `deployments/nginx/certs/` automatiquement :

```makefile
setup: check-env
    mkdir -p data/raw data/processed mlflow/artifacts ml/notebooks
    mkdir -p deployments/nginx/certs
```

Les certs eux-mêmes sont générés par le container au premier `make up`.

---

## Vérification — tests effectués en conditions réelles

### Commandes utilisées

```bash
# 1. Redirect HTTP → HTTPS
curl -s -o /dev/null -w "%{http_code} → %{redirect_url}" http://localhost

# 2. Toutes les routes HTTPS
curl -sk -o /dev/null -w "%{http_code}" https://localhost/
curl -sk -o /dev/null -w "%{http_code}" https://localhost/mlflow/
curl -sk -o /dev/null -w "%{http_code}" https://localhost/jupyter/

# 3. Headers de sécurité
curl -sk -I https://localhost/mlflow/ | grep -E "HTTP/|Strict-Transport|X-Content|X-Frame"

# 4. Redirect Jupyter suivie jusqu'au bout
curl -sk -L -o /dev/null -w "%{http_code}" https://localhost/jupyter/
```

### Résultats

| Route | Code | Interprétation |
|---|---|---|
| `http://localhost` | `301 → https://localhost/` | ✅ Redirect forcé |
| `https://localhost/` | `200` | ✅ API répond |
| `https://localhost/mlflow/` | `200` | ✅ MLflow répond |
| `https://localhost/jupyter/` | `302 → 200` | ✅ Redirect vers `/jupyter/lab` puis 200 |
| Headers HSTS, X-Content, X-Frame | présents | ✅ Sécurité active |

### Bug découvert et corrigé

La première version utilisait des blocs `upstream` résolus au démarrage de Nginx.
Cela provoquait un crash immédiat si un backend était absent :

```
[emerg] host not found in upstream "api:8000" in /etc/nginx/nginx.conf:16
```

**Correction appliquée :** remplacement des blocs `upstream` par `resolver 127.0.0.11`
+ variables `set $upstream` pour une résolution DNS à la requête (voir section Resolver).

De plus, le container Nginx en production tournait avec une image construite avant nos
modifications. Il a fallu reconstruire explicitement avec `docker compose up --build --no-deps nginx`
pour que la nouvelle config soit prise en compte.

---

## En production : passer à Let's Encrypt

Pour un vrai domaine, l'architecture Nginx reste identique — seule la source des certificats change :

1. Utiliser l'image `certbot/certbot` pour obtenir des certs Let's Encrypt.
2. Monter `/etc/letsencrypt` en volume partagé entre Certbot et Nginx.
3. Ajouter un `location /.well-known/acme-challenge/` en HTTP pour la validation ACME.
4. Planifier `certbot renew` via un cron.
