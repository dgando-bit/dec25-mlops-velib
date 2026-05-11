# shared

Package Python partagé entre tous les services du projet Vélib' MLOps.

## Installation

Depuis la racine du repo :

```bash
pip install -e shared/
```

Le mode `-e` (editable) permet à toute modification de `shared/` d'être prise en compte immédiatement par les services qui en dépendent (`api/`, `ml/`, etc.) sans réinstallation.

## Modules

- `shared.config` — configuration Pydantic (variables d'environnement validées)
- `shared.logger` — logger structuré (text/json) configurable
- `shared.utils` — helpers réutilisables (à venir)

## Variables d'environnement

Voir `.env.example` à la racine du repo pour la liste complète.
