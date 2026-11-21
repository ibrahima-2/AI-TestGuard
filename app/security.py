"""Contrôles de sécurité opérationnelle : recherche de secrets dans le dépôt."""

from __future__ import annotations

import re
from pathlib import Path

SECRET_PATTERNS = {
    "cle_aws": re.compile(r"AKIA[0-9A-Z]{16}"),
    "cle_privee": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "jeton_github": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "cle_kaggle": re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    "affectation_secret": re.compile(
        r"(?i)\b(password|passwd|secret|api_key|apikey|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "htmlcov",
    "models",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".cfg", ".ini", ".env", ".sh"}


def iter_text_files(root: str | Path):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name.startswith(".env")):
            yield path


def scan_for_secrets(root: str | Path) -> list[dict]:
    """Retourne les emplacements (fichier, ligne, type) des secrets potentiels, sans leur valeur."""
    root = Path(root)
    findings = []
    for path in iter_text_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, start=1):
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({"file": path.relative_to(root).as_posix(), "line": number, "type": name})
    return findings
