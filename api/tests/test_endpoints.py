"""Tests des endpoints FastAPI via TestClient avec modèle fixture réel."""
from __future__ import annotations

import pytest

VALID_PAYLOAD: dict = {
    "capacity": 30,
    "capacity_group": 1,
    "morning_evening_ratio": 1.05,
    "hour_sin": 0.866,
    "hour_cos": 0.5,
    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "month": 5,
    "is_peak_hour": 0,
    "is_friday_evening": 0,
    "is_monday_morning": 1,
    "is_holiday": 0,
    "is_vacation": 0,
    "apparent_temperature": 16.5,
    "temp_anomalie": 1.2,
    "weather_severity": 0,
    "is_frozen": 0,
    "is_stormy": 0,
    "lag_60min": 45.0,
    "lag_240min": 38.0,
    "lag_res_240min": -3.5,
    "lat": 48.8566,
    "lon": 2.3522,
    "hour": 8,
    "station_trend_avg": 42.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Santé & méta
# ─────────────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/docs"


def test_docs_accessible(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_model_info_returns_metadata(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "velib_fill_rate_predictor"
    assert body["alias"] == "staging"
    assert body["version"] == "1"
    assert body["n_features"] == 24


# ─────────────────────────────────────────────────────────────────────────────
# Prédiction unitaire
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_valid_returns_prediction(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "residual_predicted" in body
    assert "taux_predicted" in body
    assert body["alert_level"] in ("green", "yellow", "red")


def test_predict_taux_in_bounds(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    taux = resp.json()["taux_predicted"]
    assert 0.0 <= taux <= 100.0


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "capacity"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_extra_field_returns_422(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "unknown_field": 99})
    assert resp.status_code == 422


def test_predict_invalid_month_returns_422(client):
    resp = client.post("/predict", json={**VALID_PAYLOAD, "month": 13})
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Prédiction batch
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_batch_single_item(client):
    resp = client.post("/predict/batch", json={"items": [VALID_PAYLOAD]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_predictions"] == 1
    assert len(body["predictions"]) == 1


def test_predict_batch_multiple_items(client):
    resp = client.post("/predict/batch", json={"items": [VALID_PAYLOAD] * 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_predictions"] == 5
    assert len(body["predictions"]) == 5


def test_predict_batch_empty_returns_422(client):
    resp = client.post("/predict/batch", json={"items": []})
    assert resp.status_code == 422


def test_predict_batch_too_large_returns_422(client):
    resp = client.post("/predict/batch", json={"items": [VALID_PAYLOAD] * 2001})
    assert resp.status_code == 422


def test_predict_batch_taux_in_bounds(client):
    resp = client.post("/predict/batch", json={"items": [VALID_PAYLOAD] * 3})
    assert resp.status_code == 200
    for pred in resp.json()["predictions"]:
        assert 0.0 <= pred["taux_predicted"] <= 100.0
        assert pred["alert_level"] in ("green", "yellow", "red")


# ─────────────────────────────────────────────────────────────────────────────
# Rechargement du modèle
# ─────────────────────────────────────────────────────────────────────────────

def test_model_reload_returns_200(client):
    resp = client.post("/model/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert "new_version" in body
    assert "previous_version" in body
    assert body["model_name"] == "velib_fill_rate_predictor"
