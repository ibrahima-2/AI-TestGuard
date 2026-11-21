"""Partie VIII : sécurité opérationnelle (secrets, fichiers ignorés, journaux)."""

from __future__ import annotations

import logging

import pytest

from app.config import ROOT
from app.prediction import score_records
from app.preprocessing import DataValidationError
from app.security import scan_for_secrets

# Valeurs factices construites par concaténation pour ne pas déclencher le scanner sur ce fichier.
FAKE_SECRETS = {
    "cle_aws": "aws = " + "AKIA" + "ABCDEFGHIJKLMNOP",
    "cle_privee": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "jeton_github": "ghp" + "_" + "a" * 36,
    "cle_kaggle": '{"username": "x", "key": "' + "0123456789abcdef" * 2 + '"}',
    "affectation_secret": "password" + ' = "' + "S3cr3tValue!" + '"',
}


@pytest.mark.parametrize(("kind", "content"), FAKE_SECRETS.items(), ids=list(FAKE_SECRETS))
def test_scanner_detects_planted_secret(tmp_path, kind, content):
    (tmp_path / "config.py").write_text(f"x = 1\n{content}\n")
    findings = scan_for_secrets(tmp_path)
    assert {"file": "config.py", "line": 2, "type": kind} in findings


def test_scanner_ignores_virtualenv_and_reports_no_value(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text(FAKE_SECRETS["cle_aws"])
    (tmp_path / "ok.py").write_text("password_length = 12\n")
    assert scan_for_secrets(tmp_path) == []


def test_repository_contains_no_secret():
    assert scan_for_secrets(ROOT) == []


@pytest.mark.parametrize("pattern", ["data/*.csv", ".env", "kaggle.json", "*.joblib", ".venv/"])
def test_sensitive_files_are_git_ignored(pattern):
    assert pattern in (ROOT / ".gitignore").read_text().splitlines()


def test_rejected_request_does_not_log_transaction_values(trained_model, test_features, caplog):
    model, threshold = trained_model
    X, _ = test_features
    record = {k: float(v) for k, v in X.iloc[0].items()}
    record["Amount"] = 424242.42
    record["card_number"] = 4970101234567890
    with caplog.at_level(logging.DEBUG), pytest.raises(DataValidationError) as excinfo:
        score_records(model, [record], threshold)
    assert "4970101234567890" not in str(excinfo.value)
    assert "424242" not in str(excinfo.value)
    assert "424242" not in caplog.text
