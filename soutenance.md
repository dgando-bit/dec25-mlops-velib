# Vélib' MLOps — Document de soutenance

> Généré depuis le code source réel (branche `feat-integration`, 2026-05-21).
> Toutes les valeurs proviennent du code lu : métriques depuis `data/outputs/metrics.json`,
> hyperparamètres depuis `ml/src/models/_helpers.py`, configuration depuis `docker-compose.yml` et `.env.example`.

---

## 1. Résumé exécutif

Ce projet met en œuvre un système MLOps complet pour prédire le **taux de remplissage des stations Vélib' à Paris** — un indicateur opérationnel clé pour le rééquilibrage des vélos. Les données sont collectées automatiquement depuis HuggingFace toutes les 6 heures, nettoyées et transformées par un pipeline reproductible versionné par DVC, puis utilisées pour entraîner un modèle XGBoost. Le modèle atteint un **R² de 0,87** et une **MAE de 7,3 points de pourcentage** sur le test set. L'innovation technique centrale est une stratégie résiduelle : le modèle prédit l'écart à la tendance historique de chaque station, ce qui rend les prédictions stables indépendamment du profil de la station. Le système est orchestré par Apache Airflow (CeleryExecutor), exposé via une API FastAPI derrière un reverse proxy Nginx, suivi par Prometheus + Grafana, et présenté au jury via une interface Streamlit. La stack complète repose sur 19 services Docker Compose.

---

## 2. Architecture générale

### Couche source de données
Les snapshots d'état des stations Vélib' sont déposés à intervalles réguliers dans un repository HuggingFace Datasets (`voroman/velib-ml-data`). Le fichier consolidé `dataset_velib_raw_01.csv` contient l'historique complet. Le pipeline télécharge ce fichier via l'API `huggingface_hub`, avec retry exponentiel (tenacity) sur les erreurs transitoires (429, 5xx).

### Couche pipeline ML
Le pipeline est défini en 6 stages DVC enchaînés (`load_from_hf → make_dataset → build_features → train_model`, avec `dataviz` et `detect_drift` en branches parallèles). Chaque stage est un script Python isolé, idempotent, dont les entrées/sorties sont hashées par DVC. L'état de référence est versionné sur un remote DagsHub (S3-compatible). L'orchestration en production est gérée par Airflow (DAG `velib_ml_pipeline`).

### Couche serving
Le modèle XGBoost est enregistré dans le MLflow Model Registry sous l'alias `staging`. L'API FastAPI le charge en mémoire au démarrage via `lru_cache` et sert les prédictions sur `POST /predict` et `POST /predict/batch`. Un reload à chaud est possible sans redémarrage via `POST /model/reload`. Nginx joue le rôle de reverse proxy pour les 8 services exposés, avec rate limiting différencié (10 req/s pour l'API, 5 req/s pour les UIs internes).

### Couche observabilité
Prometheus scrape l'endpoint `/metrics` de l'API toutes les 15 secondes et collecte les métriques système via node-exporter. Grafana consomme Prometheus comme source de données unique (provisionnée automatiquement) et déclenche des alertes sur CPU > 90% et RAM > 80%. Le drift des features est détecté à chaque run de pipeline par Evidently, avec rapport HTML et JSON en sortie.

---

## 3. Composants détaillés

---

### Nginx

**Rôle** : reverse proxy unique — tout le trafic externe passe par lui avant d'atteindre un service.

**Technologie** : image `nginx:latest`, configuration via bind mount (`nginx.conf` monté en `:ro`).

**Interfaces** :
- Ports exposés (hôte → Nginx → upstream) :
  - `8080:80` → `api:8000` (API FastAPI)
  - `5000:5000` → `mlflow-server:5000`
  - `8888:8888` → `jupyter-service:8888`
  - `9090:9090` → `prometheus:9090`
  - `3000:3000` → `grafana:3000`
  - `8090:8090` → `airflow-webserver:8080`
  - `5555:5555` → `flower:5555`
  - `8501:8501` → `streamlit:8501`
  - `443:443` (TLS — mappé mais non configuré)
- Pages d'erreur personnalisées : `errors/404.html`, `errors/429.html`, `errors/50x.html`

**Dépendances** : tous les services upstreams doivent être démarrés. Le DNS Docker (`127.0.0.11`) est configuré avec `valid=10s` pour re-résoudre les upstreams.

**Présentation jury** : Nginx est notre point d'entrée unique pour tous les outils de la stack. Plutôt que d'exposer directement chaque service sur son port natif — ce qui serait difficile à sécuriser et à monitorer — toutes les requêtes passent d'abord par Nginx. Il applique du rate limiting : 10 requêtes par seconde pour l'API de prédiction, 5 par seconde pour les interfaces de monitoring. Si un client dépasse ce seuil, il reçoit une réponse HTTP 429 avec une page d'erreur personnalisée. Ce point d'entrée unique facilite aussi l'ajout futur d'authentification ou de TLS.

**Mémo technique** :

1. **Pourquoi `resolver 127.0.0.11` avec `valid=10s` ?**
   Les upstreams statiques Nginx sont résolus une seule fois au démarrage par défaut. Avec `valid=10s`, Nginx re-résout toutes les 10 secondes, ce qui permet de survivre à un redémarrage d'un service upstream sans avoir à recharger Nginx. Le resolver `127.0.0.11` est le DNS interne Docker.

2. **Quelle est la différence entre `apilimit` et `mllimit` ?**
   Deux zones de rate limiting distinctes : `apilimit` autorise 10 req/s avec burst de 20 (pour l'API de prédiction, qui peut recevoir des pics courts), `mllimit` autorise 5 req/s avec burst de 10 (pour les UIs internes — MLflow, Prometheus, Airflow, Flower). Ces limites protègent les services coûteux contre les clients non contrôlés.

3. **Pourquoi `proxy_http_version 1.1` et les headers `Upgrade` sur certains blocs ?**
   JupyterLab, Grafana, Airflow et Streamlit utilisent des WebSockets. HTTP/1.0 ne supporte pas les WebSockets ; il faut HTTP/1.1 avec les headers `Connection: upgrade` et `Upgrade: websocket` pour que le tunnel soit établi. Sans cela, les UIs interactives seraient cassées.

4. **Que se passe-t-il si un upstream est down au démarrage de Nginx ?**
   Nginx échoue au démarrage si un upstream n'est pas résolvable. C'est pourquoi `docker-compose.yml` déclare `depends_on` sur tous les services upstreams, et `make up` exécute `nginx -s reload` après un court délai pour forcer la ré-résolution DNS une fois que les services sont démarrés.

5. **Le port 443 est mappé mais non configuré — que faudrait-il faire pour activer TLS ?**
   Ajouter un `server { listen 443 ssl; }` dans `nginx.conf`, fournir un certificat et une clé privée montés en volume dans `deployments/nginx/certs/` (qui est déjà déclaré dans `docker-compose.yml`), et configurer `ssl_certificate` / `ssl_certificate_key`. Pour la démo, le port 443 est réservé mais inutilisé.

---

### FastAPI — API d'inférence

**Rôle** : servir les prédictions du modèle XGBoost et exposer les métriques Prometheus.

**Technologie** : FastAPI ≥ 0.110 + Uvicorn ≥ 0.27, Pydantic ≥ 2.5, Python 3.12-slim. Instrumenté par `prometheus-fastapi-instrumentator` (endpoint `/metrics` auto-exposé).

**Interfaces** :
- Port : `8000` (interne) / `8080` via Nginx
- Volumes montés : `./mlflow/artifacts`, `./shared`, `./api`
- Variable d'environnement clé : `MLFLOW_TRACKING_URI=http://mlflow-server:5000`, `API_MODEL_STAGE=production`

**Dépendances** : `mlflow-server` (chargement du modèle), `shared/` (config, logger).

#### Endpoint GET /health

**Rôle** : liveness check pour Docker healthcheck et monitoring.

**Réponse** :
```json
{"status": "ok", "api_version": "...", "model_loaded": true}
```
Renvoie `"degraded"` si `_MODEL_METADATA` est vide (modèle non chargé). Utilisé par le healthcheck Docker (`curl -f http://localhost:8000/health`).

#### Endpoint GET /model/info

**Rôle** : exposer les métadonnées du modèle actuellement en mémoire (nom, alias, version MLflow, run_id, framework, n_features).

**Réponse** : 503 si le modèle n'est pas chargé.

#### Endpoint POST /model/reload

**Rôle** : rechargement à chaud du modèle depuis le Registry MLflow sans redémarrage du service. Vide le `lru_cache`, recharge le modèle `staging`, met à jour les gauges Prometheus (`velib_model_r2`, `velib_model_mae`, etc.) et incrémente le compteur `velib_model_reloads_total`.

#### Endpoint POST /predict

**Rôle** : prédiction unitaire pour une station.

**Input** : `StationFeatures` — 24 features + `station_trend_avg` (25 champs Pydantic avec `extra="forbid"`).

**Output** :
```json
{"residual_predicted": -2.34, "taux_predicted": 39.66, "alert_level": "green"}
```
Règles d'alerte : `green` si 30–70%, `yellow` si 10–30% ou 70–90%, `red` si < 10% ou > 90%.

#### Endpoint POST /predict/batch

**Rôle** : prédiction pour N stations (1 à 2 000) en une requête. Plus efficace que N appels unitaires : une seule désérialisation, un seul `model.predict()` XGBoost sur un DataFrame de N lignes.

#### Endpoint GET /metrics

**Rôle** : endpoint Prometheus — exposé automatiquement par `prometheus-fastapi-instrumentator`. Contient les métriques HTTP standard + les gauges custom (`velib_model_loaded`, `velib_model_r2`, `velib_model_mae`, `velib_model_mape`, `velib_model_version`, `velib_model_reloads_total`).

**Présentation jury** : L'API est le point de contact de tous les clients qui veulent des prédictions. Elle est intentionnellement légère : elle ne calcule pas de features, ne télécharge pas de données — elle fait uniquement l'inférence. Le modèle est chargé une seule fois en mémoire au démarrage, grâce à un mécanisme de cache, donc chaque requête de prédiction ne coûte que le calcul XGBoost, sans I/O disque. Quand Airflow entraîne un nouveau modèle, il appelle `POST /model/reload` pour que l'API prenne en compte la nouvelle version sans interruption de service.

**Mémo technique** :

1. **Comment fonctionne le `lru_cache` sur le chargement du modèle ?**
   `load_model_by_alias` dans `inference.py` est décorée `@lru_cache(maxsize=4)`. L'alias (ex. `"staging"`) est la clé de cache. Le premier appel charge le modèle depuis MLflow (3–5 secondes), les suivants retournent l'objet en mémoire instantanément. `maxsize=4` permet de garder jusqu'à 4 versions/alias simultanément (utile si `"production"` et `"staging"` coexistent).

2. **Que fait `proxy_intercept_errors on` dans nginx.conf ?**
   Sans cette directive, une réponse 429 retournée par Nginx (rate limit) serait transmise telle quelle. Avec `proxy_intercept_errors on`, Nginx intercepte les erreurs de l'upstream et les redirige vers les pages d'erreur personnalisées (`errors/429.html`, etc.) définies par `error_page 429`.

3. **Pourquoi `extra="forbid"` dans `StationFeatures` ?**
   Tout champ inconnu dans le payload JSON → réponse 422 immédiate. Cela force les clients à respecter strictement le contrat API et détecte tôt les erreurs de feature engineering côté client (colonne mal nommée, feature manquante).

4. **Pourquoi l'API charge-t-elle le modèle `staging` et non `production` ?**
   La variable `API_MODEL_STAGE=production` dans `docker-compose.yml` est bien définie, mais dans `dependencies.py`, `get_model()` appelle `load_staging_model()` (hardcodé). Il y a une légère incohérence entre la variable d'env et le code : `load_staging_model()` charge toujours `staging`, indépendamment de `API_MODEL_STAGE`. C'est un point d'amélioration identifié.

5. **Que contient `_MODEL_METADATA` et à quoi sert-il ?**
   C'est un dictionnaire module-level initialisé au démarrage par `preload_model()`. Il contient `model_name`, `alias`, `version`, `run_id`, `taux_r2`, `taux_mae`, `taux_mape`, `n_features`. Il sert à deux fins : alimenter `/model/info` sans interroger MLflow à chaque appel, et vérifier que le modèle est chargé avant de servir des prédictions (si vide → 503).

---

### MLflow — Tracking et Registry

**Rôle** : stocker les expériences (runs, métriques, hyperparamètres, artefacts) et servir de registre de modèles (Model Registry avec alias).

**Technologie** : MLflow 2.22.0, backend PostgreSQL 15 (`mlflow-db`), artifacts servis en proxy HTTP via `--serve-artifacts`.

**Interfaces** :
- Port : `5000` (interne) / `5000:5000` via Nginx
- Backend store : `postgresql://user:password@mlflow-db/mlflow_db`
- Artifacts : bind mount `./mlflow/artifacts:/app/mlflow/artifacts`
- Endpoint de santé : `GET /health` (non `/api/2.0/mlflow/experiments/list` qui a été supprimé en 2.22)

**Dépendances** : `mlflow-db` (PostgreSQL), volume artifacts partagé avec `ml_training` et `api`.

**Présentation jury** : MLflow est notre centre de contrôle pour le cycle de vie du modèle. À chaque entraînement, le pipeline y enregistre automatiquement les hyperparamètres, les métriques, les graphiques de diagnostic et le modèle sérialisé. Le Model Registry permet de gérer les versions : quand un nouveau modèle est validé, on lui attribue l'alias `staging`, ce qui déclenche un reload de l'API. Le système utilise les alias MLflow — une fonctionnalité moderne — plutôt que les stages dépréciés (None/Staging/Production).

**Mémo technique** :

1. **Pourquoi `--serve-artifacts` dans la commande MLflow ?**
   Sans cette option, MLflow redirige les requêtes d'artefacts vers le backend de stockage directement (ici, le bind mount). Avec `--serve-artifacts`, MLflow agit comme proxy HTTP pour les artefacts : l'API et les autres services n'ont pas besoin de lire directement le système de fichiers. C'est plus propre en architecture micro-services, et indispensable si l'on migre vers S3 un jour.

2. **Quelle est la différence entre les stages MLflow dépréciés et les alias ?**
   Les stages (None/Staging/Production) ont été dépréciés dans MLflow 2.x. Les alias sont plus flexibles : on peut en définir autant qu'on veut, les changer atomiquement, et les requêter via `models:/model_name@alias`. Le projet utilise l'alias `staging` pour le modèle prêt à servir.

3. **Où sont stockés physiquement les artefacts (plots, model) ?**
   Dans `./mlflow/artifacts/` côté hôte, monté en bind mount dans le conteneur `mlflow-server` et dans `ml_training`. Le modèle est stocké sous `mlflow/artifacts/<experiment_id>/<run_id>/artifacts/model/`.

4. **Pourquoi deux bases PostgreSQL (mlflow-db et airflow-db) plutôt qu'une seule ?**
   Isolation des workloads : MLflow a ses propres schémas et ses propres besoins de performance (insertions fréquentes de métriques). Airflow a des patterns différents (lectures fréquentes du scheduler). Partager une base augmenterait le couplage et compliquerait les migrations de schéma.

5. **Comment récupérer les métriques d'un run depuis l'API REST MLflow ?**
   `GET /api/2.0/mlflow/runs/get?run_id=<run_id>` retourne `run.data.metrics` (liste de `{key, value, timestamp, step}`). C'est l'endpoint utilisé par `mlflow_run_metrics()` dans `api_client.py`.

---

### Airflow — Orchestration

**Rôle** : planifier et exécuter le pipeline ML toutes les 6 heures.

**Technologie** : Apache Airflow 2.10.4, CeleryExecutor, broker Redis, résultats en PostgreSQL (`airflow-db`). Services : `airflow-webserver`, `airflow-scheduler`, `airflow-worker`, `airflow-init`, `flower`.

**Interfaces** :
- UI : port `8090` via Nginx (upstream `airflow-webserver:8080`)
- Flower (monitoring Celery) : port `5555`
- Auth API : `airflow.api.auth.backend.basic_auth`
- SMTP : smtp.gmail.com:587 (pour notifications email optionnelles)

**Dépendances** : `airflow-db` (PostgreSQL), `redis` (broker), `mlflow-server` (pour les scripts ML).

#### DAG `velib_ml_pipeline`

**Schedule** : `timedelta(hours=6)` (toutes les 6 heures), `catchup=False`, `max_active_runs=1`.

**Flux** :
```
sense_new_data (PythonSensor)
    → load_from_hf (PythonOperator)
    → check_data_changed (BranchPythonOperator)
         ├─ make_dataset → build_features → train_model → notify_success
         └─ skip_training → notify_success
```

**Détails des tâches** :
- **`sense_new_data`** : `PythonSensor` en mode `reschedule`, vérifie la liste des fichiers du repo HuggingFace toutes les 30 minutes, timeout 5h. Compare l'état courant à un fichier `.last_hf_files` pour détecter un nouveau fichier.
- **`load_from_hf`** : `PythonOperator`, exécute `python -m ml.src.data.load_from_hf` via `subprocess.run()` dans le worker.
- **`check_data_changed`** : `BranchPythonOperator`, compare le SHA256 du dernier snapshot (lu dans `.snapshots.log`) à l'état sauvegardé dans `.last_dag_hash`. Si identique → `skip_training`, sinon → `make_dataset`.
- **`make_dataset` → `build_features` → `train_model`** : `PythonOperator`, chacun exécute le script ML correspondant via `subprocess`.
- **`notify_success`** : `EmailOperator`, `trigger_rule="none_failed_min_one_success"`.

**Présentation jury** : Airflow orchestre l'ensemble du cycle de vie du modèle. Toutes les 6 heures, il interroge HuggingFace pour détecter de nouvelles données. Si de nouvelles données arrivent, il déclenche le pipeline complet — téléchargement, nettoyage, feature engineering, entraînement — et enregistre le nouveau modèle dans MLflow. Si les données n'ont pas changé, il court-circuite proprement sans gaspiller des ressources. Le CeleryExecutor permet de distribuer les tâches sur des workers séparés, ce qui est la base de l'extensibilité d'Airflow.

**Mémo technique** :

1. **Pourquoi `mode="reschedule"` sur le PythonSensor plutôt que `mode="poke"` ?**
   En mode `poke`, le worker Airflow reste bloqué pendant toute la durée du sensor (jusqu'à 5h), monopolisant une slot d'exécution. En mode `reschedule`, le worker libère sa slot entre deux pokes (toutes les 30 min), permettant à d'autres tâches d'utiliser le worker. Indispensable avec CeleryExecutor pour ne pas bloquer le pool.

2. **Comment fonctionne la détection de changement SHA256 ?**
   `_check_data_changed` lit la dernière ligne de `.snapshots.log` (champ 5 = sha256_short des 8 premiers caractères). Il compare ce SHA au fichier `.last_dag_hash`. Si identique → skip. Cette approche est simple mais fragile : elle suppose que `.snapshots.log` est toujours présent et bien formaté. Si `load_from_hf` a écrit dans le log, le SHA reflète le hash du parquet produit.

3. **Pourquoi les tâches ML utilisent-elles `subprocess.run()` plutôt que d'importer directement les modules Python ?**
   Les workers Airflow ont leur propre environnement Python (`apache/airflow:2.10.4` + dépendances ML additionnelles). Exécuter via `subprocess.run()` isole proprement les processus, évite les conflits de `sys.modules` entre les imports Airflow et les imports ML, et garantit que chaque script commence avec un état propre.

4. **Que fait `airflow-init` dans docker-compose.yml ?**
   C'est un conteneur one-shot qui exécute `airflow version`. Avec les variables `_AIRFLOW_DB_MIGRATE=true` et `_AIRFLOW_WWW_USER_CREATE=true`, il initialise la base de données Airflow (migrations Alembic) et crée l'utilisateur admin. Il doit se terminer avant que `webserver` et `scheduler` démarrent (via `depends_on`).

5. **Pourquoi `email_on_failure=False` dans `DEFAULT_ARGS` alors qu'un SMTP est configuré ?**
   Le callback `on_failure_callback` loggue l'échec en JSON structuré, mais `email_on_failure=False` désactive l'envoi automatique d'emails sur échec. Seule la tâche `notify_success` envoie un email (succès). C'est un choix délibéré pour éviter le spam en cas de flapping réseau, mais cela signifie qu'un échec pipeline silencieux peut passer inaperçu si personne ne surveille les logs.

---

### Pipeline DVC — Stage 1 : `load_from_hf`

**Rôle** : télécharger tous les fichiers CSV depuis HuggingFace et produire un unique parquet brut consolidé.

**Technologie** : `huggingface_hub~=0.36`, `tenacity>=8.2`, `pandas 2.3.3`, compression zstd.

**Sortie** : `data/raw/velib_snapshot_latest.parquet` + `.snapshots.log`

**Présentation jury** : Ce premier stage est notre connecteur avec la source de données. Il liste les fichiers du repo HuggingFace (au format `dataset_velib_raw_NN.csv`), les télécharge, les concatène, supprime les doublons (station_id × datetime), et écrit un parquet compressé. Pour garantir la robustesse, les appels HuggingFace sont enveloppés dans un mécanisme de retry exponentiel : en cas d'erreur 429 (rate limit) ou 5xx, il retente jusqu'à 3 fois.

**Mémo technique** :

1. **Pourquoi `force_download=True` dans les settings par défaut ?**
   Configuré via `HF_FORCE_DOWNLOAD=true` dans `.env`. Cela ignore le cache local HuggingFace et re-télécharge le fichier à chaque exécution. Justifié tant que le dataset tient en un seul fichier `_01.csv` : on veut toujours la version la plus récente. À désactiver (`false`) quand le dataset dépassera ~2 Go ou quand HF aura créé un fichier `_02.csv`.

2. **Comment fonctionne le retry exponentiel ?**
   `_retry_decorator()` construit un décorateur tenacity avec `wait_exponential(multiplier=5.0, min=5.0, max=120.0)` et `stop_after_attempt(3)`. Seules les erreurs retryables déclenchent un retry : `HfHubHTTPError` avec status 429 ou 5xx, `ConnectionError`, `TimeoutError`. Les 404 et 403 échouent immédiatement.

3. **Pourquoi le SHA256 dans `.snapshots.log` est-il seulement 8 caractères ?**
   Suffisant pour identifier un snapshot dans le contexte du projet (probabilité de collision négligeable sur un historique de quelques milliers de runs). L'objectif est la traçabilité humaine rapide, pas la sécurité cryptographique.

---

### Pipeline DVC — Stage 2 : `make_dataset`

**Rôle** : nettoyer le snapshot brut — coercer les types, calculer le taux de remplissage, filtrer les lignes invalides, exclure les stations sans variance.

**Entrée** : `data/raw/velib_snapshot_latest.parquet`
**Sortie** : `data/interim/velib_cleaned_latest.parquet` + `.cleaning.log`

**4 étapes de nettoyage** :
1. Coercition booléens (`is_renting`, `is_holiday`, `is_vacation`) → bool propre
2. Calcul `total_capacity` et `taux = num_bikes_available / total_capacity * 100`
3. Filtres qualité : `is_renting==True`, `capacity > 0`, `total_capacity > 0`, `taux ∈ [0, 100]`
4. Exclusion des stations avec variance < 0.5 (configuré via `MIN_VARIANCE` dans `.env`)

**Contrôle qualité** : cohérence avec `capacity_status` du collector (tolérance 0.5 pp) — warning seulement, ne bloque pas le pipeline.

**Mémo technique** :

1. **Pourquoi exclure les stations à faible variance ?**
   Une station qui affiche toujours le même taux (ex. une station hors service ou saturée en permanence) ne peut pas être apprise par XGBoost. Elle polluerait le dataset sans valeur ajoutée. Le seuil `min_variance=0.5` est configurable via `.env`.

2. **Pourquoi calculer `taux = num_bikes_available / total_capacity * 100` plutôt qu'utiliser `capacity_status` directement ?**
   Le collector produit `capacity_status` par arrondi, ce qui introduit des imprécisions. Recalculer depuis les composantes brutes garantit la cohérence et la précision flottante. Le script vérifie ensuite la cohérence avec `capacity_status` (tolérance 0.5 pp) pour détecter des bugs upstream.

---

### Pipeline DVC — Stage 3 : `dataviz`

**Rôle** : produire un rapport HTML Plotly interactif pour l'exploration visuelle du dataset.

**Sortie** : `data/outputs/plots/dataviz_report.html` (`cache: false` dans dvc.yaml — non poussé sur le remote DVC)

Ce stage est documentaire. Il dépend de `data/raw` ET `data/interim` pour montrer avant/après nettoyage. Les 7 graphiques incluent : distribution du taux, séries temporelles, heatmaps heures × jours de la semaine, etc.

---

### Pipeline DVC — Stage 4 : `build_features`

**Rôle** : 9 étapes de feature engineering, split temporel 80/20, export parquet train/test.

**Entrée** : `data/interim/velib_cleaned_latest.parquet`
**Sorties** : `train_preprocessed.parquet`, `test_preprocessed.parquet`, `stations_geo.parquet`

**9 étapes** :
1. Features temporelles : `hour`, `dow`, `month`, encodage cyclique (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`)
2. Profil station : `capacity_group` (0=<15, 1=15-24, 2=25-39, 3=≥40 places)
3. Features météo : `weather_severity` (0–4), `is_frozen`, `is_stormy`
4. Encodage booléens calendaires : `is_holiday`, `is_vacation` → int8
5. Lags temporels : `lag_60min`, `lag_240min` via `merge_asof` (pas de fuite → alignement exact sur l'horodatage)
6. Split temporel : quantile 1 − test_size (= 0.20) → les 80% les plus anciens en train, les 20% les plus récents en test
7. Post-split features (calculées sur train seulement pour éviter la fuite) : `morning_evening_ratio`, `temp_anomalie`
8. `station_trend_avg` : calculé sur train (agrégat station × dow × hour, ou + month si ≥ 90 jours d'historique)
9. Cible résiduelle : `residual_target = taux - station_trend_avg`, `lag_res_240min = lag_240min - station_trend_avg`

**Présentation jury** : L'étape clé ici est le split temporel et le calcul de `station_trend_avg`. Le split est temporel et non aléatoire : les données les plus récentes constituent le test set, ce qui simule un vrai déploiement. La tendance historique (`station_trend_avg`) est calculée uniquement sur le train set, puis jointe au test set par lookup — jamais calculée sur le test, ce qui évite toute fuite de données.

**Mémo technique** :

1. **Pourquoi un split temporel par quantile plutôt qu'un split aléatoire ?**
   Avec un split aléatoire, des observations du test set pourraient être temporellement antérieures à des observations du train set. Le modèle apprendrait implicitement sur le futur, ce qui gonflerait artificiellement les métriques. Le quantile temporel garantit que le test set est strictement postérieur au train set.

2. **Comment fonctionne le fallback adaptatif de `station_trend_avg` ?**
   Si le train set couvre moins de 90 jours, la clé d'agrégation est `(station_id, dow, hour)`. Si ≥ 90 jours, la clé est `(station_id, dow, hour, month)` pour capturer la saisonnalité mensuelle. Pour les stations sans observation dans le test set (nouvelles stations), un fallback par `(dow, hour)` global est appliqué.

3. **Pourquoi `morning_evening_ratio` est-il calculé après le split ?**
   Ce ratio (`taux_moyen_matin / taux_moyen_soir`) est une statistique descriptive de chaque station. S'il était calculé avant le split, il inclurait des observations du test set pour estimer la distribution du train, ce qui constituerait une fuite. En le calculant sur le train seulement, on évite ce problème.

---

### Pipeline DVC — Stage 5 : `train_model`

**Rôle** : entraîner XGBoost sur la cible résiduelle, logger dans MLflow, promouvoir en alias `staging`.

**Entrées** : `train_preprocessed.parquet`, `test_preprocessed.parquet`
**Sorties** : `data/outputs/metrics.json`, 3 plots PNG, modèle dans MLflow Registry

**Pipeline sklearn** : `SimpleImputer(strategy="median") → XGBRegressor(hyperparams)`

**Hyperparamètres XGBoost** (depuis Optuna, figés) :
| Paramètre | Valeur |
|---|---|
| n_estimators | 232 |
| max_depth | 6 |
| learning_rate | 0.122 |
| subsample | 0.937 |
| colsample_bytree | 0.524 |
| colsample_bylevel | 0.562 |
| min_child_weight | 21 |
| reg_alpha (L1) | 1.40 |
| reg_lambda (L2) | 1.47 |
| gamma | 0.776 |
| tree_method | hist |
| n_jobs | -1 |

**Métriques (run `ebaf8e1e`, 2026-05-20)** :
| Métrique | Valeur |
|---|---|
| taux_r2 | **0.870** |
| taux_mae | **7.28 pp** |
| taux_rmse | 10.72 pp |
| taux_mape_pct | 31.79% |
| residual_r2 | 0.785 |
| residual_mae | 7.29 pp |

**Présentation jury** : Le script d'entraînement est entièrement automatisé et reproductible. Il configure MLflow, ouvre un run, logge tous les hyperparamètres, entraîne XGBoost, évalue sur le test set, génère trois graphiques de diagnostic (importances des features, distribution des résidus, prédictions vs réalité), enregistre le modèle dans le Registry et pose automatiquement l'alias `staging`. Après cet appel, l'API peut charger la nouvelle version sans redémarrage.

**Mémo technique** :

1. **Pourquoi `SimpleImputer` en amont de XGBoost alors que XGBoost gère nativement les NaN ?**
   XGBoost gère les NaN par son propre mécanisme (learnt direction). L'imputer est là par cohérence avec l'API sklearn Pipeline (pour `model.predict()` uniforme) et pour protéger les pipelines futurs si un préprocesseur upstream ne gère pas les NaN. C'est une ceinture et bretelles.

2. **Comment `infer_signature` affecte-t-il l'utilisation future du modèle ?**
   `mlflow.models.signature.infer_signature(X_train, y_pred)` capture les types et les dimensions d'entrée/sortie. Lors du chargement (`mlflow.sklearn.load_model`), MLflow peut valider les inputs contre cette signature. Sans signature, les erreurs de features ne seraient détectées qu'à l'exécution de `.predict()`.

3. **Les métriques sont sur le résidu ET sur le taux reconstruit — pourquoi les deux ?**
   Les métriques résiduelles reflètent la qualité brute du modèle (ce qu'il apprend). Les métriques sur le taux reconstruit reflètent la valeur métier (ce que voit l'opérateur). Un R² résiduel de 0.785 vs taux R² de 0.870 montre que `station_trend_avg` contribue à une grande part de la variance expliquée — le modèle corrige efficacement autour de la tendance.

4. **Que signifie un taux_mape de 31.79% ?**
   La MAPE (Mean Absolute Percentage Error) est calculée en ignorant les stations avec `taux_réel < 1%` (division par zéro). 31.79% est élevé en valeur absolue mais attendu sur un taux de remplissage : une erreur de 7.3 pp sur un taux réel de 23% donne une MAPE de ~32%. C'est la limite inhérente de prédire des pourcentages proches de 0 ou 100.

5. **Pourquoi `tree_method="hist"` ?**
   L'algorithme `hist` (Histogram-based gradient boosting) est plus rapide que `exact` sur de grands datasets (approximation des splits). Compatible avec CPU et GPU. Pour 1 492 stations × milliers de timestamps, c'est le choix approprié.

---

### Pipeline DVC — Stage 6 : `detect_drift`

**Rôle** : mesurer le drift des 24 features entre le train set (référence) et le test set (courant).

**Technologie** : `evidently==0.4.36` (version figée, 0.4.37+ incompatible avec numpy==2.2.6), `DataDriftPreset`.

**Sorties** :
- `data/outputs/drift/drift_report.html` (rapport Evidently interactif, `cache: false`)
- `data/outputs/drift/drift_metrics.json` : `{n_features, n_drifted, drift_share, dataset_drift, drift_alert}`

**Seuil d'alerte** : `DRIFT_SHARE_THRESHOLD = 0.30` — si > 30% des features ont drifté, un warning est loggué. Ce seuil est **informatif uniquement** — il ne bloque pas la promotion du modèle.

**Limite méthodologique importante** : le drift mesuré compare `train_preprocessed.parquet` (80% les plus anciens) à `test_preprocessed.parquet` (20% les plus récents) du **même snapshot**. Ce n'est donc pas un proxy de dérive en production — c'est principalement la dérive saisonnière inhérente au split temporel (les features `month`, `apparent_temperature`, `weather_severity` driftent mécaniquement entre hiver et printemps).

**Mémo technique** :

1. **Pourquoi `evidently==0.4.36` et pas la dernière version ?**
   Evidently 0.4.37+ impose `numpy<2.1`. Le projet utilise `numpy==2.2.6` (requis par `mlflow>=2.18+`). Pin à 0.4.36 pour éviter le conflit de dépendances.

2. **Pourquoi le drift mesuré ne reflète pas une dérive de production réelle ?**
   Un vrai monitoring de drift en production comparerait un snapshot HF récent (données fraîches post-entraînement) à une fenêtre glissante de référence. Ici, les deux jeux de données sont issus du même snapshot — le drift mesuré est donc un artefact du split temporel, pas un signal de dérive opérationnel.

---

### Streamlit — Application de démonstration

**Rôle** : interface de présentation au jury — 5 pages navigables, status live des services, visualisations interactives, et formulaire de prédiction.

**Technologie** : Streamlit ≥ 1.35, Plotly ≥ 5.20, Python 3.12-slim + Docker CLI installé. Volume Docker socket monté (`/var/run/docker.sock`) pour le pytest runner intégré.

**Interfaces** :
- Port : `8501` / `8501` via Nginx
- Variables d'env : `API_URL`, `MLFLOW_URL`, `PROMETHEUS_URL`, `GRAFANA_URL`, `AIRFLOW_URL`, `NGINX_URL`, `HOST_PROJECT_ROOT`
- Volume `${HOST_PROJECT_ROOT}:${HOST_PROJECT_ROOT}` pour lancer `docker compose run --rm api pytest`

**5 pages** :
1. **Présentation & Dataviz** (`01_Presentation_Dataviz.py`) : KPIs modèle, 7 graphiques Plotly (portage de `dataviz.py`), continuité HF, schémas SVG architecture
2. **Background MLOps** (`00_Background_MLOps.py`) : status live des 5 services, graphique continuité HF (bar chart 288 commits/j), schéma containers SVG zoomable (svg-pan-zoom), métriques modèle live, liens vers UIs natives
3. **Tests et Validations** (`02_Validation.py`) : tests fonctionnels par onglet (API, MLflow, Prometheus, Grafana, Airflow, Nginx 404/429) + pytest runner
4. **Prédiction** (`03_Prediction.py`) : 15 stations hardcodées, contexte temporel/météo, `POST /predict`, gauge Plotly, badge alerte green/yellow/red
5. **Les prochaines étapes** (`04_Prochaines_Etapes.py`) : limitations, roadmap, trajectoire cloud

**Présentation jury** : L'application Streamlit est notre interface de démonstration. En une page, on voit le statut de tous les services en temps réel — si l'API répond, si MLflow est accessible, si Prometheus collecte des métriques. On peut envoyer une vraie requête de prédiction : sélectionner une station, un contexte temporel et météo, et voir la prédiction s'afficher avec son niveau d'alerte. Le pytest runner intégré permet de lancer la suite de tests depuis l'interface, ce qui facilite la validation en démo.

**Mémo technique** :

1. **Pourquoi le Docker socket est-il monté dans Streamlit ?**
   Le pytest runner lance `docker compose run --rm api pytest api/tests/` depuis le conteneur Streamlit. Pour exécuter des commandes Docker depuis un conteneur, il faut accès au socket Docker de l'hôte. Le groupe `DOCKER_GID` est ajouté (via `group_add`) pour que l'utilisateur Streamlit ait les droits sur le socket.

2. **Les stations affichées dans la page Prédiction sont-elles réelles ?**
   Les 15 stations dans `utils/stations.py` ont des coordonnées et des capacités réelles, mais leurs `station_trend_avg` sont des valeurs hardcodées approximatives (ex. 55.0 pour Gare du Nord). Ce ne sont pas des moyennes calculées depuis le vrai historique. En production, cette valeur serait lue depuis `stations_geo.parquet`.

3. **Comment fonctionne `hf_dataset_commits()` dans `api_client.py` ?**
   Il pagine l'API `GET /api/datasets/{id}/commits/main?limit=1000&p=N` jusqu'à 20 pages max (20 000 commits potentiels). Pour chaque commit, il extrait la date (`c["date"][:10]`). Le résultat est une liste de `datetime.date` triée, utilisée pour construire le bar chart de continuité (commits par jour).

4. **Pourquoi `st.cache_data` sur `_load_svg()` ?**
   Les SVGs sont des fichiers statiques lus depuis le système de fichiers. Sans cache, Streamlit les relirait à chaque re-render de la page (toutes les interactions). `@st.cache_data` mémorise le résultat de la lecture disque pour la session.

5. **Quelle est la limite de la page Prédiction ?**
   Les features `lag_60min` et `lag_240min` sont saisies manuellement par l'utilisateur — dans un système de production, elles seraient calculées automatiquement depuis l'historique réel. De même, `apparent_temperature` et `temp_anomalie` ne viennent pas d'une vraie API météo.

---

### Prometheus

**Rôle** : base de données de séries temporelles (TSDB) pour stocker les métriques de l'API et du système.

**Technologie** : `prom/prometheus:latest`, retention 15 jours.

**Configuration** (`prometheus.yml`) :
- `scrape_interval: 15s` — collecte toutes les 15 secondes
- `evaluation_interval: 15s` — évalue les règles d'alerte toutes les 15 secondes

**3 jobs de scrape** :
| Job | Target | Endpoint |
|---|---|---|
| `velib-api` | `api:8000` | `/metrics` |
| `prometheus` | `localhost:9090` | auto |
| `node-exporter` | `node-exporter:9100` | auto |

**Métriques custom API** :
- `velib_model_loaded` (Gauge) — 1 si modèle chargé
- `velib_model_version`, `velib_model_r2`, `velib_model_mae`, `velib_model_mape` (Gauges)
- `velib_model_reloads_total` (Counter)
- Métriques HTTP standard : `http_requests_total`, `http_request_duration_seconds` (via `prometheus-fastapi-instrumentator`)

**Présentation jury** : Prometheus est notre base de métriques. Toutes les 15 secondes, il collecte l'état de l'API — combien de requêtes ont été traitées, quelle est leur latence, si le modèle est chargé, quelles sont ses métriques de performance. Il collecte aussi les métriques système (CPU, RAM, disque) via node-exporter. Ces données alimentent Grafana en temps réel.

**Mémo technique** :

1. **Qu'est-ce qu'une TSDB et pourquoi Prometheus plutôt qu'un SQL ?**
   Une Time Series Database stocke des couples `(timestamp, valeur)` par métrique, optimisée pour les lectures temporelles et les agrégations (sum over time, rate). Prometheus utilise un format binaire interne très compact. Une base SQL classique serait beaucoup moins efficace pour des milliards de points horodatés.

2. **Comment fonctionne `--storage.tsdb.retention.time=15d` ?**
   Prometheus supprime automatiquement les données de plus de 15 jours. Cela borne l'espace disque utilisé. Pour un projet de démonstration, 15 jours est largement suffisant. En production, on augmenterait ou on utiliserait Thanos/Cortex pour le stockage long terme.

3. **Pourquoi scraper `localhost:9090` dans le job `prometheus` ?**
   Prometheus se scrappe lui-même pour exposer ses propres métriques internes (nombre de séries, latence des requêtes, mémoire). Cela permet de monitorer la santé de Prometheus depuis... Prometheus.

4. **Comment `prometheus-fastapi-instrumentator` fonctionne-t-il ?**
   `Instrumentator().instrument(app).expose(app, endpoint="/metrics")` ajoute un middleware FastAPI qui mesure chaque requête HTTP. Il expose les métriques Prometheus sur `/metrics` au format exposition Prometheus. C'est une bibliothèque standard qui n'impacte pas les performances de manière significative.

5. **Pourquoi Prometheus est-il configuré comme `depends_on: api` dans docker-compose.yml ?**
   Si l'API n'est pas démarrée, le premier scrape de Prometheus échouera. Bien que Prometheus soit tolérant aux targets down (il retente), `depends_on` garantit que l'API est au moins lancée avant que Prometheus commence à scraper.

---

### Grafana

**Rôle** : visualisation des métriques Prometheus et alerting.

**Technologie** : `grafana/grafana:latest`, provisionnée automatiquement.

**Interfaces** :
- Port : `3000` via Nginx
- Volume persistent : `grafana_data:/var/lib/grafana`
- Provisioning : `./deployments/grafana/provisioning/` monté en `:ro`

**Provisionnement automatique** :
- **Datasource** : Prometheus (`http://prometheus:9090`), `uid: prometheus`, `isDefault: true`
- **Dashboards** : 3 fichiers JSON (`Infra-dashboard.json`, `velib_dashboard.json`) + `dashboard.yml`

**2 règles d'alerte** (`alerts.yaml`) :
| Alerte | Condition | Durée | Contact |
|---|---|---|---|
| CPU Usage High | `100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100 > 90` | 5 min | `empty` |
| Memory Usage High | `(MemTotal - MemFree) / MemTotal * 100 > 80` | 2 min | `empty` |

**Point faible** : le `receiver: empty` signifie qu'aucune notification n'est envoyée quand une alerte se déclenche. Les alertes sont visibles dans l'UI Grafana mais aucun canal (email, Slack, PagerDuty) n'est configuré.

**Présentation jury** : Grafana est notre tableau de bord d'observabilité. Il consomme toutes les métriques collectées par Prometheus et les affiche en graphiques temps réel. On peut voir l'utilisation CPU et mémoire du serveur, la latence de l'API, le nombre de requêtes par seconde, et les métriques du modèle (R², MAE) qui changent à chaque re-entraînement. Les alertes se déclenchent automatiquement si le CPU dépasse 90% pendant 5 minutes.

**Mémo technique** :

1. **Qu'est-ce que le provisionnement automatique Grafana ?**
   Grafana peut lire des fichiers YAML/JSON au démarrage pour créer automatiquement datasources, dashboards et règles d'alerte — sans passer par l'UI. Les fichiers dans `./deployments/grafana/provisioning/` sont montés en lecture seule. Au redémarrage du conteneur, la configuration est recréée automatiquement.

2. **Pourquoi `receiver: empty` dans les alertes ?**
   Le contact point `empty` est un receveur nul — il accepte les notifications mais ne les transmet nulle part. C'est un placeholder en attente d'une vraie intégration (email SMTP, webhook Slack). En l'état, les alertes sont visibles dans l'UI Grafana mais aucun humain n't en est notifié en dehors de Grafana.

3. **Comment la query CPU est-elle calculée ?**
   `100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)` : le taux de CPU idle sur 5 minutes est calculé par node-exporter, on soustrait de 100% pour obtenir le CPU utilisé. `rate()` lisse les fluctuations courtes.

4. **Comment le dashboard charge-t-il les métriques du modèle ?**
   Les gauges `velib_model_r2`, `velib_model_mae`, etc. sont des métriques Prometheus custom exposées par l'API. Le dashboard Grafana les requête via PromQL. Ces valeurs se mettent à jour à chaque reload de modèle (`POST /model/reload`).

5. **Que faudrait-il faire pour activer les notifications d'alerte ?**
   Dans `alerts.yaml`, remplacer `receiver: empty` par un contact point configuré. Dans l'UI Grafana : `Alerting → Contact points → New contact point` (email, Slack, webhook). Ou provisionner le contact point via un fichier YAML dans `provisioning/alerting/`.

---

### PostgreSQL — mlflow-db

**Rôle** : backend de stockage pour le serveur MLflow (métadonnées des runs, expériences, versions de modèles).

**Technologie** : `postgres:15`, volume persistent `mlflow_db_data`.

**Configuration** :
- Database : `mlflow_db` (via `MLFLOW_DB` dans `.env`)
- User : `MLFLOW_USER`
- Password : `MLFLOW_PASSWORD`
- Healthcheck : `pg_isready -U ${MLFLOW_USER} -d ${MLFLOW_DB}`

**Présentation jury** : C'est la base de données de MLflow. Elle stocke toutes les métadonnées des expériences — quels hyperparamètres ont été testés, quelles métriques ont été obtenues, quel modèle correspond à quel run. Les fichiers binaires (modèles, plots) ne sont pas dans cette base mais dans le volume `mlflow/artifacts/`.

---

### PostgreSQL — airflow-db

**Rôle** : backend de métadonnées pour Airflow (état des DAGs, runs, logs de tâches, résultats Celery).

**Technologie** : `postgres:15`, volume persistent `airflow_db_data`.

**Configuration** :
- Database : `airflow`
- User : `airflow`
- Password : `AIRFLOW_DB_PASSWORD`
- Healthcheck : `pg_isready -U airflow`
- Restart : `always`

**Double usage** :
- `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` → métadonnées Airflow
- `AIRFLOW__CELERY__RESULT_BACKEND` → résultats des tâches Celery

---

### Redis

**Rôle** : broker de messages Celery — transmet les tâches entre le scheduler Airflow et les workers.

**Technologie** : `redis:latest`, healthcheck `redis-cli ping`, restart `always`.

**Configuration** : `AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0`

**Note de sécurité** : `:@redis:6379/0` signifie que Redis n'est pas protégé par mot de passe. Acceptable dans un réseau Docker isolé (`mlops-net`), non acceptable en production exposée.

**Présentation jury** : Redis sert de file d'attente entre le scheduler Airflow (qui décide quelles tâches lancer) et les workers Celery (qui les exécutent). Quand le scheduler déclenche une tâche, il envoie un message à Redis. Le worker disponible le consomme, exécute la tâche, et publie son résultat dans la base Airflow. Flower (port 5555) permet de visualiser l'état de cette file en temps réel.

**Mémo technique** :

1. **Pourquoi Redis comme broker plutôt que la base PostgreSQL directement ?**
   En mode `LocalExecutor` (mono-nœud), PostgreSQL peut servir de queue. Mais avec `CeleryExecutor` (multi-workers), Redis est plus adapté : il est conçu pour la messagerie haute performance, supporte les pub/sub, et évite de charger la base Airflow avec des lectures polling intensives.

2. **Que stocke Redis vs la base PostgreSQL Airflow ?**
   Redis stocke les messages en transit (tâches en attente d'exécution). PostgreSQL stocke l'historique durable (état final des tâches, logs). Redis peut perdre des données à un redémarrage (pas de persistance par défaut) — ce n'est pas gênant car les tâches non consommées seront re-soumises par le scheduler.

---

### node-exporter

**Rôle** : exporter les métriques système de l'hôte (CPU, RAM, disque, réseau) vers Prometheus.

**Technologie** : `prom/node-exporter:latest`, restart `unless-stopped`.

**Configuration** :
```yaml
pid: host
volumes:
  - /proc:/host/proc:ro
  - /sys:/host/sys:ro
  - /:/rootfs:ro
command:
  - '--path.procfs=/host/proc'
  - '--path.sysfs=/host/sys'
  - '--collector.filesystem.mount-points-exclude=...'
```

**Présentation jury** : node-exporter est un agent léger qui lit les statistiques système de l'hôte Linux et les expose au format Prometheus. Il donne à Grafana des métriques précises sur le CPU (par cœur), la RAM, l'espace disque, et le réseau. Sans lui, on ne verrait que les métriques applicatives de l'API.

**Mémo technique** :

1. **Pourquoi `pid: host` ?**
   Sans cette option, le conteneur a son propre espace de nommage PID et ne voit que ses propres processus. Avec `pid: host`, il partage l'espace PID de l'hôte et peut mesurer le CPU de tous les processus — y compris les autres conteneurs et le système d'exploitation hôte.

2. **Pourquoi monter `/proc`, `/sys` et `/` en lecture seule ?**
   `/proc` expose les statistiques CPU, mémoire et processus du noyau Linux. `/sys` expose les informations hardware (disque, réseau). `/` (rootfs) permet les métriques de système de fichiers. Montés en `:ro`, le conteneur ne peut pas modifier ces pseudo-systèmes de fichiers — pure lecture.

---

## 4. Modèle ML

### Architecture du modèle

Le modèle est un **XGBoost Regressor** (Gradient Boosting à base d'arbres) enveloppé dans un Pipeline sklearn avec un `SimpleImputer(strategy="median")` en amont.

**Stratégie résiduelle** : le modèle ne prédit pas directement `taux_remplissage`. Il prédit le **résidu** `taux - station_trend_avg`. La reconstruction se fait : `taux_prédit = station_trend_avg + résidu_prédit`, clippé dans [0, 100].

Justification : `station_trend_avg` capture la composante lente et prévisible du taux (tendance calendaire station × heure × jour de semaine). Le modèle n'apprend que la correction à apporter à cette tendance. Cela réduit la variance de la cible (moins d'amplitude à prédire), ce qui améliore les métriques et stabilise l'entraînement.

### Métriques de performance

**Dernier run** (`ebaf8e1e`, 2026-05-20T14:50:01Z) :

| Métrique | Valeur | Interprétation |
|---|---|---|
| **taux_r2** | **0.870** | 87% de la variance du taux expliquée par le modèle |
| **taux_mae** | **7.28 pp** | Erreur absolue moyenne de 7.3 points de pourcentage |
| taux_rmse | 10.72 pp | Sensible aux grosses erreurs (>2× le MAE) |
| taux_mape_pct | 31.79% | Élevé (attendu sur taux proches de 0) |
| residual_r2 | 0.785 | 78.5% de la variance du résidu expliquée |
| residual_mae | 7.29 pp | Quasi-identique au taux_mae (logique : le résidu et le taux partagent la même magnitude) |
| Stations couvertes | 1 492 (CLAUDE.md) | — |

### Stratégie de feature engineering

**24 features** réparties en 6 groupes :

| Groupe | Features | Rôle |
|---|---|---|
| Profil station | `capacity`, `capacity_group`, `morning_evening_ratio` | Caractéristiques intrinsèques de la station |
| Temporel cyclique | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month` | Continuité cyclique (ex. minuit ↔ 23h) |
| Flags temporels | `is_peak_hour`, `is_friday_evening`, `is_monday_morning` | Événements de forte demande |
| Calendaire | `is_holiday`, `is_vacation` | Perturbations prévisibles |
| Météo | `apparent_temperature`, `temp_anomalie`, `weather_severity`, `is_frozen`, `is_stormy` | Impact direct sur l'utilisation |
| Lags | `lag_60min`, `lag_240min`, `lag_res_240min` | Auto-corrélation temporelle |
| Géographie | `lat`, `lon` | Localisation spatiale (quartier) |
| Temporel brut | `hour` | Redondance intentionnelle avec sin/cos |

**Note** : `station_trend_avg` n'est PAS une feature du modèle — l'inclure serait une fuite triviale (la cible est `taux - station_trend_avg`).

### Gestion du drift

Evidently (`DataDriftPreset`) compare les 24 features entre train (référence) et test (courant) du même snapshot. Le seuil d'alerte est 30% de features driftées. **Limitation documentée dans le code** : ce drift reflète la saisonnalité du split temporel, pas une dérive production réelle.

### Présentation jury (pitch 5 phrases)

Nous avons choisi XGBoost pour sa robustesse sur des données tabulaires structurées avec peu de préparation. L'innovation principale n'est pas l'algorithme mais la stratégie résiduelle : plutôt que de prédire un taux de remplissage brut qui varie énormément d'une station à l'autre et d'une heure à l'autre, nous prédisons l'écart à la tendance historique de chaque station. Cela réduit la variance de la cible et rend le modèle plus facile à entraîner et à interpréter. Avec un R² de 0,87 et une MAE de 7,3 points de pourcentage sur 1 492 stations parisiennes, le modèle est suffisamment précis pour guider les décisions de rééquilibrage opérationnel. Les hyperparamètres ont été optimisés par Optuna lors d'une phase exploratoire préalable, puis figés pour garantir la reproductibilité des runs suivants.

### Mémo technique (5 questions)

1. **Pourquoi XGBoost et pas un réseau de neurones (LSTM, Transformer) ?**
   Les séries temporelles Vélib' ont une structure tabulaire forte : chaque observation est indépendante une fois les lags calculés. Les LSTM sont adaptés aux séquences longues avec dépendances temporelles complexes. Ici, les lags à 60 et 240 min capturent l'essentiel de l'auto-corrélation. XGBoost est plus rapide à entraîner, plus interprétable (importance des features), et son sur-apprentissage est mieux contrôlé sur ce type de données. C'est aussi le choix habituel dans les benchmarks de ML tabulaire.

2. **Pourquoi l'encodage cyclique (sin/cos) plutôt que `hour` brut ?**
   `hour` brut traite l'heure comme une variable linéaire : 23 et 0 sembleraient distants de 23 unités alors qu'ils sont adjacents. L'encodage `sin(2π·h/24)` / `cos(2π·h/24)` préserve la continuité du cycle : minuit est proche de 23h dans l'espace de features. Le raw `hour` est conservé en plus comme redondance intentionnelle.

3. **Comment station_trend_avg est-elle calculée et comment évite-t-elle la fuite de données ?**
   Elle est calculée exclusivement sur le train set : pour chaque combinaison `(station_id, dow, hour)` (ou `+ month` si ≥ 90 jours d'historique), on calcule la moyenne du taux. Cette table est ensuite jointe au test set par lookup. Les stations du test set non vues en train (nouvelles stations) reçoivent un fallback sur `(dow, hour)` global. Aucune observation du test n'entre dans le calcul.

4. **Pourquoi `min_child_weight=21` (valeur haute) ?**
   Un `min_child_weight` élevé force XGBoost à exiger beaucoup d'observations dans chaque feuille pour effectuer un split. Cela régularise fortement le modèle, réduit le sur-apprentissage, et améliore la généralisation. Avec 1 492 stations × des milliers de timestamps, les feuilles peuvent rapidement devenir trop spécifiques à des stations rares.

5. **Pourquoi clipper le taux prédit entre [0, 100] ?**
   XGBoost est un régresseur non contraint — rien ne l'empêche de prédire `station_trend_avg + résidu = 105%`. Physiquement, un taux > 100% ou < 0% est impossible. Le clip garantit que les sorties de l'API restent dans un domaine valide, indépendamment des valeurs extrêmes du résidu prédit.

---

## 5. Flux de données de bout en bout

**De la source HuggingFace à la prédiction servie**

Un collecteur externe (HuggingFace Space, non couvert ici) dépose régulièrement un fichier CSV consolidé dans le dataset `voroman/velib-ml-data`. Ce fichier contient des snapshots d'état de toutes les stations Vélib' (station_id, datetime, num_bikes_available, capacity, is_renting, coordonnées GPS, is_holiday, is_vacation, apparent_temperature, etc.).

Le **stage 1** (`load_from_hf`) est déclenché par Airflow. Il interroge l'API HuggingFace pour lister les fichiers du repo, télécharge `dataset_velib_raw_01.csv` (avec retry exponentiel), le lit avec `pandas.read_csv(parse_dates=["datetime"])`, concatène (si plusieurs fichiers), supprime les doublons `(station_id, datetime)`, trie, et écrit `data/raw/velib_snapshot_latest.parquet` (compression zstd).

Le **stage 2** (`make_dataset`) lit ce parquet brut, coerce les types booléens, calcule `total_capacity = num_docks_available + num_bikes_available` et `taux = num_bikes_available / total_capacity * 100`, filtre les lignes invalides (stations fermées, capacités nulles, taux hors [0, 100]), exclut les stations sans variance (seuil 0.5), et écrit `data/interim/velib_cleaned_latest.parquet`.

Le **stage 4** (`build_features`) applique 9 étapes de feature engineering sur les données nettoyées. Il calcule les encodages cycliques, les profils de station, les features météo, et les lags temporels via `merge_asof`. Il effectue le split temporel 80/20 par quantile. Sur le train seulement, il calcule `station_trend_avg` et les features post-split (`morning_evening_ratio`, `temp_anomalie`). Enfin il dérive la cible résiduelle `residual_target = taux - station_trend_avg`. Les trois parquets train/test/stations_geo sont écrits dans `data/processed/`.

Le **stage 5** (`train_model`) charge les parquets, ouvre un run MLflow, entraîne le pipeline `SimpleImputer → XGBRegressor(24 features → résidu)`, évalue les métriques (R², MAE, MAPE sur résidu et taux), logue tout dans MLflow (hyperparamètres, métriques, 3 plots, le modèle), enregistre dans le Registry sous `velib_fill_rate_predictor@staging`, et exporte `metrics.json`.

L'**API FastAPI** charge au démarrage (via `lru_cache`) le modèle `velib_fill_rate_predictor@staging` depuis MLflow. À chaque requête `POST /predict`, elle reçoit les 24 features précalculées + `station_trend_avg`, appelle `model.predict()`, reconstruit `taux = station_trend_avg + résidu_prédit`, applique les règles d'alerte, et retourne `{residual_predicted, taux_predicted, alert_level}`.

---

## 6. Flux d'une requête d'inférence

**De la requête HTTP au JSON de réponse**

Un client envoie `POST http://localhost:8080/predict` avec un payload JSON contenant les 25 champs définis dans `StationFeatures`.

**Étape 1 — Nginx (rate limiting)** : Nginx reçoit la requête sur le port 8080 (mappé sur son port 80 interne). Il vérifie la zone `apilimit` : si le client a envoyé plus de 10 requêtes par seconde (burst de 20), Nginx retourne immédiatement HTTP 429 avec la page `errors/429.html`. Sinon, il ajoute les headers `X-Real-IP`, `X-Forwarded-For`, `X-Forwarded-Proto` et transmet la requête à `http://api_backend` (= `api:8000`).

**Étape 2 — Uvicorn / FastAPI (parsing)** : Uvicorn reçoit la requête ASGI. FastAPI route vers `POST /predict`. Le middleware prometheus-fastapi-instrumentator commence à mesurer la latence.

**Étape 3 — Validation Pydantic** : FastAPI instancie `StationFeatures(**body)`. Pydantic valide les 25 champs : types, ranges (`ge=0`, `le=100`, etc.), extra fields (`extra="forbid"`). Si un champ est absent ou invalide → HTTP 422 immédiat.

**Étape 4 — Vérification modèle** : `_MODEL_METADATA` est vérifié non-vide. Si vide (mode dégradé) → HTTP 503.

**Étape 5 — Injection du modèle (Depends)** : `Depends(get_model)` appelle `get_model()` → `load_staging_model()` → `load_model_by_alias("staging")` avec `@lru_cache`. Premier appel : chargement MLflow (3–5s). Appels suivants : retour instantané du cache.

**Étape 6 — Préparation du DataFrame** : `_predict_single()` extrait `station_trend_avg` du payload, construit un DataFrame Pandas à 1 ligne et 24 colonnes (FEATURES_FINAL), sans `station_trend_avg`.

**Étape 7 — Inférence XGBoost** : `predict_with_confidence(features=df, station_trend_avg=pd.Series([trend]), model=model)` appelle `model.predict(features[FEATURES_FINAL])`. Le pipeline sklearn : SimpleImputer (médiane sur NaN résiduels) → XGBRegressor.predict() → résidu prédit.

**Étape 8 — Reconstruction et alerte** : `taux_prédit = np.clip(station_trend_avg + résidu_prédit, 0, 100)`. Règles d'alerte : si `taux < 10` ou `> 90` → `"red"`, si `taux < 30` ou `> 70` → `"yellow"`, sinon → `"green"`.

**Étape 9 — Sérialisation** : FastAPI sérialise `PredictionResponse(residual_predicted, taux_predicted, alert_level)` en JSON. Les valeurs flottantes sont arrondies à 2 décimales.

**Étape 10 — Prometheus** : À la fin de la requête, le middleware prometheus-fastapi-instrumentator enregistre la latence et incrémente `http_requests_total{method="POST", handler="/predict", status="200"}`.

**Réponse retournée au client** :
```json
{"residual_predicted": -2.34, "taux_predicted": 39.66, "alert_level": "green"}
```

---

## 7. Points faibles et axes d'amélioration

### 1. Contact point Grafana vide — alertes sans destinataire

**Constat dans le code** : `alerts.yaml`, `notification_settings: receiver: empty` pour les deux règles (CPU >90%, RAM >80%).

**Impact** : Les alertes se déclenchent dans l'UI Grafana mais aucun humain n'est notifié (pas d'email, pas de Slack, pas de PagerDuty).

**Amélioration** : Configurer un contact point réel dans `provisioning/alerting/`. Pour un déploiement production : webhook Slack ou PagerDuty avec escalade.

---

### 2. Stations Streamlit hardcodées avec tendances approximatives

**Constat** : `streamlit/app/utils/stations.py` contient 15 stations avec des `station_trend_avg` fixes (ex. `55.0` pour Gare du Nord). Ces valeurs ne proviennent pas du vrai historique.

**Impact** : Les prédictions dans la page Prédiction sont valides algorithmiquement mais basées sur une tendance inexacte — elles ne reflètent pas l'état réel de la station.

**Amélioration** : Lire `station_trend_avg` depuis `data/processed/stations_geo.parquet` produit par le pipeline, qui contient les vraies moyennes calculées sur le dataset d'entraînement.

---

### 3. Drift mesuré sur split temporel — pas un proxy production

**Constat** : `detect_drift.py`, commentaire lignes 39–44 : "current = test_preprocessed.parquet, issu du même snapshot. Le drift reflète la dérive saisonnière intrinsèque au split."

**Impact** : `drift_share` dans `drift_metrics.json` est un signal bruité. Les features `month`, `apparent_temperature`, `weather_severity` driftent mécaniquement entre les périodes couverte par train et test (saisons différentes).

**Amélioration** : Comparer un nouveau snapshot HF (données post-entraînement) à la fenêtre d'entraînement. Ajouter une tâche Airflow qui télécharge un snapshot récent et le compare au train de référence.

---

### 4. Redis sans authentification

**Constat** : `AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0` — le `:` avant `@` signifie pas de mot de passe.

**Impact** : Acceptable dans le réseau interne `mlops-net` isolé par Docker. Non acceptable si Redis est jamais exposé au réseau hôte ou à Internet.

**Amélioration** : Ajouter `requirepass` dans la configuration Redis et mettre à jour `BROKER_URL` avec le mot de passe.

---

### 5. Port 443 mappé mais TLS non configuré

**Constat** : `nginx` ports `- "443:443"`, mais aucun `server { listen 443 ssl; }` dans `nginx.conf`. Le répertoire `deployments/nginx/certs/` est monté mais vide.

**Impact** : Tout le trafic entre le navigateur du client et l'application est en clair (HTTP). Les tokens JupyterLab, les credentials Airflow, les prédictions — tout transite non chiffré.

**Amélioration** : Générer un certificat auto-signé (démo) ou Let's Encrypt (production), configurer `ssl_certificate` et `ssl_certificate_key` dans nginx.conf, et rediriger HTTP → HTTPS.

---

### 6. `API_MODEL_STAGE` ignoré par le code

**Constat** : `docker-compose.yml` déclare `API_MODEL_STAGE=production`, mais `dependencies.py:53` appelle `load_staging_model()` (hardcodé `"staging"`), ignorant la variable d'env.

**Impact** : L'API charge toujours le modèle `staging`, jamais `production`. La séparation staging/production est illusoire.

**Amélioration** : `get_model()` devrait lire `settings.api_model_stage` et appeler `load_model_by_alias(settings.api_model_stage)`.

---

### 7. Lags Streamlit saisis manuellement

**Constat** : `03_Prediction.py` — l'utilisateur saisit `lag_60min` et `lag_240min` via des sliders. Ces valeurs ne viennent pas d'un historique réel.

**Impact** : Démo fonctionnelle mais irréaliste. Un utilisateur qui saisit `lag_60min=50` et `lag_240min=50` obtient une prédiction cohérente avec ces inputs, mais ces valeurs ne correspondent pas à l'état réel de la station à l'instant T.

**Amélioration** : Intégrer un appel à l'API temps réel Vélib' pour récupérer l'état actuel de la station et calculer les lags automatiquement.

---

### 8. `email_on_failure=False` dans le DAG

**Constat** : `velib_pipeline.py:DEFAULT_ARGS`, `email_on_failure: False`. Si une tâche Airflow échoue (load_from_hf, train_model), aucun email n'est envoyé.

**Impact** : Un échec silencieux du pipeline peut passer inaperçu si personne ne consulte l'UI Airflow.

**Amélioration** : Passer à `email_on_failure: True` et configurer correctement `AIRFLOW_SMTP_*` dans `.env`. Ou utiliser des alertes Grafana sur des métriques Airflow (nb de DAGs en échec).

---

## 8. Glossaire technique

**TSDB (Time Series Database)** : base de données optimisée pour le stockage de séries temporelles — des paires (timestamp, valeur) associées à un label. Prometheus en est un exemple. Les lectures agrégées sur des fenêtres temporelles (rate, sum_over_time) y sont beaucoup plus rapides qu'en SQL classique.

**lru_cache** : `functools.lru_cache` — décorateur Python qui mémorise les résultats d'une fonction pour des arguments donnés. LRU = Least Recently Used : quand le cache est plein, l'entrée la moins récemment utilisée est supprimée. Dans ce projet, utilisé pour mettre en cache le modèle MLflow chargé depuis le Registry.

**Scrape (Prometheus)** : action de collecter les métriques en faisant un `GET /metrics` sur une cible. Prometheus est un système pull — c'est lui qui interroge les services, pas les services qui lui poussent des métriques.

**Alias MLflow (Registry)** : un alias est un pointeur nommé vers une version spécifique d'un modèle dans le MLflow Registry (ex. `staging → version 7`). Contrairement aux stages dépréciés, on peut créer autant d'alias qu'on veut, les déplacer atomiquement, et les interroger via `models:/model_name@alias`.

**CeleryExecutor** : mode d'exécution Airflow où les tâches sont distribuées à des workers via un broker de messages (ici Redis). Permet plusieurs workers en parallèle sur différentes machines. Alternative au LocalExecutor (mono-machine, sans broker).

**Drift (data drift)** : changement statistique dans la distribution des features entre deux périodes. Si les données de production ressemblent de moins en moins aux données d'entraînement, le modèle peut se dégrader. Evidently détecte ce drift via des tests statistiques (Kolmogorov-Smirnov pour les features continues, chi2 pour les catégorielles).

**Résiduel (stratégie de prédiction)** : plutôt que de prédire directement une valeur `y`, on prédit l'écart `y - baseline`, où `baseline` est une estimation simple (ici, la tendance historique `station_trend_avg`). La cible résiduelle a une amplitude réduite, ce qui facilite l'apprentissage et améliore les métriques.

**Feature engineering** : processus de transformation des données brutes en features numériques exploitables par un modèle ML. Inclut : encodage cyclique, lags temporels, agrégations, flags binaires, normalisation. C'est souvent la partie la plus importante du travail ML.

**Reverse proxy** : serveur qui reçoit les requêtes à la place d'un ou plusieurs serveurs cibles, les transmet, et retourne les réponses. Il permet de centraliser le rate limiting, le TLS, les logs d'accès, et l'équilibrage de charge. Nginx est le reverse proxy de ce projet.

**Rate limiting** : mécanisme qui limite le nombre de requêtes qu'un client peut envoyer par unité de temps. Protège contre les attaques DoS et les clients trop agressifs. Nginx implémente un algorithme token bucket (`limit_req_zone`).

**DVC (Data Version Control)** : outil de versioning de données et de pipelines ML. Les fichiers de données (parquets lourds) sont stockés sur un remote (ici DagsHub), tandis que leurs hashes sont versionnés dans Git. `dvc repro` réexécute uniquement les stages dont les inputs ont changé.

**MLflow Model Registry** : composant MLflow qui stocke les versions de modèles avec leurs métadonnées. Permet de promouvoir des versions en alias (staging, production), de les comparer, et de les charger par nom symbolique plutôt que par chemin absolu.

**Lifespan (FastAPI)** : gestionnaire de contexte asynchrone (`@asynccontextmanager`) qui remplace les anciens event handlers `@app.on_event("startup")`. Le code avant `yield` s'exécute au démarrage de l'application, le code après `yield` au shutdown.

**Bind mount** : montage d'un répertoire ou fichier de l'hôte directement dans un conteneur Docker (`./local/path:/container/path`). Différent d'un volume nommé (géré par Docker) : les changements sont immédiatement visibles des deux côtés.

**Parquet** : format de fichier de données colonaire binaire (Apache). Beaucoup plus compact et plus rapide à lire qu'un CSV pour des opérations analytiques (lecture de colonnes spécifiques, filtres pushdown). Ce projet utilise la compression zstd.

**Celery** : framework Python de files de tâches distribuées, utilisé ici comme executor par Airflow. Un scheduler envoie des tâches à un broker (Redis), des workers Celery les consomment et exécutent le code Python correspondant.

**Flower** : interface web de monitoring pour Celery. Affiche les workers actifs, les tâches en cours et terminées, les statistiques de performance. Accessible sur le port 5555.

**node-exporter** : agent Prometheus pour les métriques système Linux. Lit `/proc` et `/sys` et les expose au format Prometheus sur le port 9100.

**SimpleImputer** : transformateur sklearn qui remplace les valeurs manquantes (NaN) par une statistique (moyenne, médiane, mode). Dans ce projet, `strategy="median"` protège contre les outliers dans les nouvelles stations sans historique de lag.

**Reconstruction de cible** : opération inverse de la stratégie résiduelle. `taux_prédit = station_trend_avg + résidu_prédit`, clippé dans [0, 100]. Cette opération est déterministe et ne nécessite pas le modèle.

**COMPOSE_PROJECT_NAME** : variable d'environnement Docker Compose qui préfixe tous les conteneurs et volumes créés. Avec `COMPOSE_PROJECT_NAME=velib`, les conteneurs s'appellent `velib-api-1`, `velib-mlflow-server-1`, etc. Changer cette valeur isole complètement deux instances de la stack sur la même machine.

**Pipeline sklearn** : séquence de transformateurs et d'un estimateur final (`Pipeline([("imputer", ...), ("model", ...)])`). `.fit(X, y)` appelle `fit_transform` sur chaque transformateur puis `fit` sur le modèle final. `.predict(X)` appelle `transform` sur chaque transformateur puis `predict`. Garantit que train et test passent exactement par les mêmes transformations.

**infer_signature** : `mlflow.models.signature.infer_signature(X, y_pred)` capture les schémas d'entrée et sortie d'un modèle. Stockée dans le Registry, elle permet à MLflow de valider les inputs avant l'inférence et de documenter le contrat du modèle.

**PythonSensor** : opérateur Airflow qui appelle une fonction Python à intervalle régulier (`poke_interval`) jusqu'à ce qu'elle retourne `True`. En mode `reschedule`, il libère le worker entre les pokes. Ici utilisé pour détecter de nouveaux fichiers sur HuggingFace.

**BranchPythonOperator** : opérateur Airflow qui retourne l'id de la prochaine tâche à exécuter (parmi plusieurs branches). Ici, retourne `"make_dataset"` ou `"skip_training"` selon si les données ont changé.

---

*Document de référence — branche `feat-integration`, run MLflow `ebaf8e1e` (2026-05-20)*
