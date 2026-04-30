from fastapi import FastAPI

app = FastAPI(
    title="Vélib MLOps API",
    description="API d'inférence pour les prédictions Vélib",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
def root():
    return {"message": "Vélib API — voir /docs pour la documentation"}