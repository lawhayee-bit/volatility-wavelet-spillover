from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import hashlib


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_datestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
