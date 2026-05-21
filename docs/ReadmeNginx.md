# Nginx + SSL — Explication simple

## Le problème de départ

Les trois services du projet (l'API, MLflow, Jupyter) étaient accessibles directement
sur des ports différents (`8000`, `5000`, `8888`) et **sans chiffrement** — les données
transitaient en clair sur le réseau.

---

## Ce qu'on a mis en place

**Nginx** est un programme qui sert de **porte d'entrée unique**. Au lieu d'avoir trois
portes séparées, tout le monde passe par une seule porte sécurisée, et c'est Nginx qui
redirige ensuite vers le bon service en interne.

---

## Étape 1 — Les règles de Nginx (`nginx.conf`)

J'ai écrit les règles que Nginx doit suivre :

- **Si quelqu'un arrive en HTTP** (non sécurisé) → on le renvoie automatiquement vers
  HTTPS (sécurisé). Comme une boutique qui dit "entrée par l'autre côté".
- **Si quelqu'un arrive sur `/mlflow/`** → on l'envoie vers le serveur MLflow.
- **Si quelqu'un arrive sur `/jupyter/`** → on l'envoie vers Jupyter.
  Jupyter a besoin d'un traitement spécial (**WebSocket**) car il doit maintenir une
  connexion permanente ouverte pour faire tourner du code en temps réel.
- **Tout le reste** → ça va vers l'API.

J'ai aussi ajouté des **en-têtes de sécurité** : ce sont de petites instructions que
Nginx envoie au navigateur pour lui dire "n'accepte que le HTTPS",
"ne laisse pas ce site s'afficher dans une iframe", etc.

---

## Étape 2 — Le certificat SSL

Le SSL, c'est ce qui permet de **chiffrer** la communication entre le navigateur et le
serveur. Pour ça, il faut un **certificat** — une carte d'identité numérique qui prouve
que le serveur est bien celui qu'il prétend être.

Au lieu de générer le certificat à la main sur chaque machine (compliqué), j'ai fait en
sorte que **Nginx le génère lui-même automatiquement au premier démarrage**.

Concrètement : un petit script (`entrypoint.sh`) vérifie au lancement du container si un
certificat existe déjà. S'il n'existe pas (premier lancement), il en crée un. Sinon, il
le réutilise. Résultat : zéro manipulation manuelle.

Le certificat est **auto-signé** : c'est toi-même qui le crées, sans passer par une
autorité officielle. C'est parfait pour du développement local. En production, on
utiliserait Let's Encrypt (gratuit, reconnu par tous les navigateurs).
Le navigateur affichera un avertissement "connexion non privée" — c'est normal, il suffit
de l'accepter.

---

## Étape 3 — Empaqueter Nginx dans Docker (`Dockerfile`)

Pour que Nginx tourne avec les autres services dans Docker, j'ai créé une recette
(`Dockerfile`) qui dit :
1. Prends l'image officielle Nginx
2. Installe OpenSSL (l'outil qui génère les certificats)
3. Copie notre configuration et notre script de démarrage

---

## Étape 4 — Brancher Nginx au projet (`docker-compose.yml`)

J'ai ajouté Nginx comme un service supplémentaire. Il est configuré pour :
- **Écouter** les ports 80 (HTTP) et 443 (HTTPS)
- **Sauvegarder** les certificats générés dans un dossier sur ta machine
  (pour ne pas en recréer à chaque redémarrage)
- **Démarrer après** les autres services (sinon il essaierait de rediriger vers des
  services qui n'existent pas encore)

J'ai aussi ajouté une ligne à Jupyter pour lui dire qu'il est accessible sous `/jupyter/`.
Sans ça, ses fichiers CSS et JavaScript seraient introuvables et l'interface serait cassée.

---

## Étape 5 — Protéger la clé privée

J'ai ajouté la clé privée dans le `.gitignore` pour qu'elle ne soit **jamais envoyée
sur GitHub**. Une clé privée qui fuite sur GitHub, c'est comme publier le mot de passe
de ta maison.

---

## Pour lancer le projet (n'importe quelle machine)

```bash
git clone <repo>
cd dec25-mlops-velib
cp .env.example .env   # remplir les variables
make setup
make build
make up
```

C'est tout — les certificats SSL sont générés automatiquement.

---

## Résultat final

```
Avant :  navigateur → API       (port 8000, pas de chiffrement)
         navigateur → MLflow    (port 5000, pas de chiffrement)
         navigateur → Jupyter   (port 8888, pas de chiffrement)

Après :  navigateur → Nginx (port 443, HTTPS chiffré)
                          ├── /          → API
                          ├── /mlflow/   → MLflow
                          └── /jupyter/  → Jupyter
```

Une seule porte, sécurisée, qui distribue le trafic en interne.

---

## Bug trouvé et corrigé lors des tests

### Le problème

La première version de la config utilisait des blocs `upstream` — une façon classique de
déclarer les adresses des services en haut du fichier :

```
upstream api     → api:8000
upstream mlflow  → mlflow-server:5000
upstream jupyter → jupyter-service:8888
```

Le souci : Nginx essaie de **résoudre ces noms au démarrage**. Si un service n'est pas encore
lancé (ou si on teste Nginx seul, hors Docker Compose), Nginx plante immédiatement avec :

```
host not found in upstream "api:8000"
```

C'est comme si tu voulais appeler quelqu'un mais que ton téléphone cherchait le numéro
dans l'annuaire au moment où tu l'allumes, pas au moment où tu appelles.

### La solution

On a remplacé les blocs `upstream` par un **résolveur DNS à la demande** :

```
resolver 127.0.0.11   ← adresse du DNS interne de Docker
set $upstream api:8000  ← variable résolue à chaque requête, pas au démarrage
```

Maintenant Nginx cherche l'adresse du service uniquement quand une vraie requête arrive.
Si un service est temporairement down, Nginx reste debout et renvoie une erreur propre
plutôt que de crasher.

---

## Vérification — ce qu'on a testé

Après le déploiement, on a vérifié que chaque route répondait correctement :

| Test | Résultat attendu | Résultat obtenu |
|---|---|---|
| `http://localhost` (port 80) | Redirect 301 → HTTPS | ✅ OK |
| `https://localhost/` | API répond (200) | ✅ OK |
| `https://localhost/mlflow/` | MLflow répond (200) | ✅ OK |
| `https://localhost/jupyter/` | Jupyter redirige (302 → 200) | ✅ OK |
| Headers de sécurité | HSTS, X-Content, X-Frame présents | ✅ OK |

> Le **302** sur `/jupyter/` est normal : Jupyter redirige automatiquement vers
> `/jupyter/lab`, qui répond bien en 200.
