# 👤 Auteur
# 📦 Structure du projet
```
.
├── api/                    # Service de prédiction (ex: FastAPI ou Flask)
│   ├── Dockerfile          # Image pour exposer le modèle via une API
│   └── requirements.txt    # Dépendances spécifiques au service web
├── data/                   # Gestion des données (souvent dans le .gitignore)
│   ├── processed/          # Données nettoyées prêtes pour l'entraînement
│   └── raw/                # Données brutes récupérées de l'API Vélib
├── deployments/            # Infrastructure et Ops
│   ├── nginx/              # Serveur proxy (sécurité, SSL, routage)
│   │   ├── certs/          # Certificats HTTPS
│   │   └── Dockerfile
│   └── prometheus/         # Monitoring des métriques (latence, erreurs)
│       └── prometheus.yml
├── docker-compose.yml      # Orchestration de tous les services en local
├── main.py                 # Point d'entrée principal (orchestrateur)
├── Makefile                # Raccourcis de commandes (ex: make train, make deploy)
├── ml/                     # Le cœur du Machine Learning
│   ├── src/
│   │   ├── features/       # Scripts de transformation de variables
│   │   │   └── engineering.py
│	│	├── models/         # Entrainer les modeles
│   │   └── main.py
│   ├── Dockerfile          # Image pour l'entraînement ou le processing
│   └── requirements.txt
├── mlflow/                 # Tracking des expériences et registre de modèles
│   ├── Dockerfile
│   └── scripts/            # Scripts pour initialiser la DB ou le stockage S3
├── README.md               # Documentation du projet
└── shared/                 # Code partagé entre l'API et le module ML
    ├── pyproject.toml      # Configuration des outils (Black, Isort, Flake8)
    └── python/
        ├── config.py       # Variables d'environnement (API keys, ports)
        ├── logger.py       # Configuration centralisée des logs
        └── utils/
            └── helpers.py  # Fonctions utilitaires génériques
     
```