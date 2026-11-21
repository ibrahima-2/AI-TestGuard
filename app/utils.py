"""Fonctions utilitaires : empreintes, versions, sérialisation JSON."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from app import __version__
from app.config import ROOT


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Empreinte SHA-256 d'un fichier."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """Versions de l'interpréteur et des bibliothèques qui influencent les résultats."""
    return {
        "app": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def git_commit(root: Path = ROOT) -> str:
    """Identifiant court du commit courant, ou « non versionné »."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "non versionné"
    return result.stdout.strip() or "non versionné"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def to_jsonable(obj: Any) -> Any:
    """Convertit récursivement en types JSON standards (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
