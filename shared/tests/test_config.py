"""Tests des validateurs Pydantic de shared/shared/config.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.config import Settings


# ─────────────────────────────────────────────────────────────────────────────
# log_level
# ─────────────────────────────────────────────────────────────────────────────

def test_log_level_valid_values():
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        s = Settings(log_level=level)
        assert s.log_level == level


def test_log_level_case_insensitive():
    s = Settings(log_level="info")
    assert s.log_level == "INFO"


def test_log_level_invalid_raises():
    with pytest.raises(ValidationError):
        Settings(log_level="VERBOSE")


# ─────────────────────────────────────────────────────────────────────────────
# log_format
# ─────────────────────────────────────────────────────────────────────────────

def test_log_format_valid():
    for fmt in ("text", "json"):
        s = Settings(log_format=fmt)
        assert s.log_format == fmt


def test_log_format_case_insensitive():
    s = Settings(log_format="JSON")
    assert s.log_format == "json"


def test_log_format_invalid_raises():
    with pytest.raises(ValidationError):
        Settings(log_format="xml")


# ─────────────────────────────────────────────────────────────────────────────
# hf_repo
# ─────────────────────────────────────────────────────────────────────────────

def test_hf_repo_valid_format():
    s = Settings(hf_repo="user/repo")
    assert s.hf_repo == "user/repo"


def test_hf_repo_extracts_from_hf_url():
    s = Settings(hf_repo="https://huggingface.co/datasets/voroman/velib-ml-data")
    assert s.hf_repo == "voroman/velib-ml-data"


def test_hf_repo_invalid_no_slash_raises():
    with pytest.raises(ValidationError):
        Settings(hf_repo="notarepo")


def test_hf_repo_invalid_too_many_slashes_raises():
    with pytest.raises(ValidationError):
        Settings(hf_repo="a/b/c")


# ─────────────────────────────────────────────────────────────────────────────
# test_size
# ─────────────────────────────────────────────────────────────────────────────

def test_test_size_valid_boundary_zero():
    s = Settings(test_size=0.0)
    assert s.test_size == 0.0


def test_test_size_valid_boundary_one():
    s = Settings(test_size=1.0)
    assert s.test_size == 1.0


def test_test_size_out_of_bounds_raises():
    with pytest.raises(ValidationError):
        Settings(test_size=1.1)
    with pytest.raises(ValidationError):
        Settings(test_size=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# mlflow_tracking_uri default
# ─────────────────────────────────────────────────────────────────────────────

def test_mlflow_tracking_uri_default():
    s = Settings()
    assert s.mlflow_tracking_uri == "http://mlflow-server:5000"


def test_mlflow_tracking_uri_overridable():
    s = Settings(mlflow_tracking_uri="http://localhost:5000")
    assert s.mlflow_tracking_uri == "http://localhost:5000"
