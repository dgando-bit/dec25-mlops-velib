# Guide de contribution

## setup Nginx



### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et démarré
- `make` disponible (inclus sur Mac/Linux ; sur Windows : Git Bash ou WSL)

### Étapes

```bash
# 1. Cloner le repo
git clone <repo-url>
cd dec25-mlops-velib

# 2. Créer le fichier d'environnement (Nginx n'en a pas besoin, mais make check-env le vérifie)
cp .env.example .env

# 3. Créer les dossiers nécessaires
make setup

# 4. Démarrer Nginx (build de l'image + démarrage + vérification automatique)
make up-nginx
```

`make up-nginx` fait tout : il construit l'image Docker, démarre le container, génère
le certificat SSL automatiquement, puis lance `make check-nginx` pour confirmer que
tout fonctionne.

### Vérifier que ça marche

```bash
make check-nginx
```

Tu dois voir 5 checkmarks verts :

```
══ Vérification Nginx ══

[1/5] Container                       ✓ dec25-mlops-nginx est démarré
[2/5] HTTP → HTTPS redirect (port 80) ✓ 301 Redirect
[3/5] HTTPS → API (/)                 ✓ 200 OK
[4/5] HTTPS → MLflow (/mlflow/)       ✓ 200 OK
[5/5] HTTPS → Jupyter (/jupyter/)     ✓ 200 OK
```

> Les cases 3, 4 et 5 nécessitent que les autres services soient démarrés.
> Si tu testes Nginx seul, tu verras 502 (normal) — Nginx fonctionne quand même.

### URLs d'accès

| Service | URL |
|---------|-----|
| API     | https://localhost/ |
| MLflow  | https://localhost/mlflow/ |
| Jupyter | https://localhost/jupyter/ |

Le navigateur affiche un avertissement "connexion non sécurisée" — c'est attendu avec
un certificat auto-signé. Clique sur "Continuer quand même" (ou "Advanced → Proceed").

### Commandes utiles

```bash
make up-nginx       # Build + démarrer Nginx + vérification
make check-nginx    # Tester toutes les routes (rapport vert/rouge)
make logs-nginx     # Voir les logs Nginx en temps réel
make down           # Arrêter tous les services
```

### Si quelque chose ne marche pas

Repère le message d'erreur que tu vois et suis la solution correspondante.

---

####  `make: command not found`

**Ce que ça veut dire :** `make` n'est pas installé sur ta machine.

**Sur Windows :**
1. Ouvre **Git Bash** (installé avec Git for Windows — [télécharger ici](https://git-scm.com/download/win))
2. Dans Git Bash, retape la commande. `make` est inclus dedans.

> Si tu n'as pas Git Bash, tu peux aussi activer **WSL** (Windows Subsystem for Linux)
> dans les paramètres Windows, puis relancer dans un terminal Ubuntu.

---

####  `Cannot connect to the Docker daemon`

```
error during connect: Get "http://.../info": dial tcp: connection refused
```

**Ce que ça veut dire :** Docker Desktop n'est pas démarré.

**Solution :**
1. Cherche **Docker Desktop** dans le menu Démarrer et ouvre-le
2. Attends que l'icône baleine dans la barre des tâches soit stable 
3. Retape ta commande

---

#### `port is already allocated` ou `address already in use`

```
Error: Ports are not available: listen tcp 0.0.0.0:443: bind: address already in use
```

**Ce que ça veut dire :** un autre programme utilise déjà le port 80 ou 443 (souvent un autre Nginx).

**Solution :**
```bash
# Voir quel container utilise déjà le port
docker ps

# Arrêter tous les containers du projet
make down

# Relancer
make up-nginx
```

Si le problème persiste, c'est un programme Windows (pas Docker) qui bloque le port.
Redémarre ta machine et relance.

---

#### ⚠️ `check-nginx` affiche 502 sur les cases 3, 4 ou 5

```
[3/5] HTTPS → API (/)           Code inattendu : 502 (API démarrée ?)
```

**Ce que ça veut dire :** Nginx fonctionne, mais le service qu'il essaie de joindre (API, MLflow ou Jupyter) n'est pas démarré. C'est **normal si tu testes Nginx seul**.

**Solution :** Pour tout démarrer d'un coup :
```bash
make up
```

Puis relance `make check-nginx` tout doit passer au vert.

---

####  Le navigateur affiche "Votre connexion n'est pas privée"

C'est **normal et attendu** avec notre certificat auto-signé. Ce n'est pas un bug.

**Sur Chrome :**
Clique sur **"Paramètres avancés"** en bas de la page → puis **"Continuer vers localhost (dangereux)"**

**Sur Firefox :**
Clique sur **"Avancé…"** → puis **"Accepter le risque et continuer"**

**Sur Edge :**
Clique sur **"Détails"** → puis **"Accéder à localhost (non sécurisé)"**

---

####  Le container s'arrête immédiatement après le démarrage

```bash
make logs-nginx
```

Si tu vois une ligne contenant `[emerg]` ou `unknown directive`, c'est une erreur dans
la config Nginx. Note le message exact et contacte moi.

---

####  `/jupyter/` affiche une page blanche ou cassée

**Ce que ça veut dire :** Jupyter n'a peut-être pas été redémarré avec la bonne config.

**Solution :**
```bash
# Redémarrer uniquement Jupyter
docker compose up -d --no-deps jupyter-service

# Puis vérifier
make check-nginx
```

### Ce que fait Nginx dans ce projet

```
Internet / Browser
       │
   port 80 (HTTP) ──── redirect 301 → HTTPS
       │
   port 443 (HTTPS)
       │  certificat auto-signé (généré au premier démarrage)
       │
   ┌───┴──────────────────────────────────┐
   │  /           → API        (port 8000) │
   │  /mlflow/    → MLflow     (port 5000) │
   │  /jupyter/   → Jupyter    (port 8888) │
   └──────────────────────────────────────┘
          réseau interne Docker (mlops-net)
```

Pour la doc technique complète : va dans le dossier docs chemin du fichier[`docs/nginx-ssl-setup.md`](docs/nginx-ssl-setup.md)
Pour l'explication simple : aussi dans le dossier docs chemin du fichier [`docs/ReadmeNginx.md`](docs/ReadmeNginx.md)

---

## Autres contributions

Pour contribuer au reste du projet (ML, API, DVC), consulte le `README.md`.
