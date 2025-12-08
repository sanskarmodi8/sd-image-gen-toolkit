"""Runtime downloader for external LoRA adapters.

If a LoRA file is not found locally under `assets/loras/`,
we download it from the original source URL.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import requests

from sdgen.config.lora_urls import LORA_URLS
from sdgen.config.paths import ASSETS_ROOT
from sdgen.utils.logger import get_logger

logger = get_logger(__name__)

LORA_DIR: Path = ASSETS_ROOT / "loras"


def ensure_lora_dir() -> None:
    """Create lora directory if missing."""
    LORA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path, chunk: int = 8192) -> None:
    """Stream download a file to destination path."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    logger.info("Downloaded LoRA: %s", dst.name)


def ensure_loras() -> None:
    """Download missing LoRA weights at runtime."""
    ensure_lora_dir()

    for filename, url in LORA_URLS.items():
        path = LORA_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            logger.info("LoRA exists: %s", filename)
            continue

        logger.info("Downloading LoRA: %s", filename)
        try:
            download_file(url, path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to download LoRA %s: %s", filename, exc)
            # cleanup partial file
            if path.exists():
                try:
                    os.remove(path)
                except Exception:
                    pass
