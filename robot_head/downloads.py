"""Model files that are fetched on first use and kept in models/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .config import PROJECT_DIR

MODELS_DIR = PROJECT_DIR / "models"


def fetch(url: str, path: Path, description: str) -> Path:
    """Download ``url`` to ``path`` unless it is already there."""
    if not path.exists():
        print(f"Downloading {description} (first run only)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    return path
