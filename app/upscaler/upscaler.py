"""Unified upscaler interface.

Chooses between:
- NCNN RealESRGAN (fastest, works on NVIDIA/AMD/Intel)
- Future SD-upscaler backend
"""

from __future__ import annotations

from PIL import Image

from app.upscaler.realesrgan import NCNNUpscaler
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Upscaler:
    """Unified high-level upscaling wrapper."""

    def __init__(self, scale: float = 2.0, prefer: str = "ncnn"):
        """Initialize the upscaler with given backend preference."""
        logger.info(f"Upscaler initializing (prefer={prefer}, scale={scale})")

        self.engine = None

        if prefer in ("ncnn", "auto"):
            try:
                self.engine = NCNNUpscaler(scale=scale)
                logger.info("Using NCNN RealESRGAN engine.")
                return
            except Exception as err:
                logger.warning(f"NCNN RealESRGAN init failed: {err}")

        raise RuntimeError("No valid upscaler engine available.")

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale the given image."""
        return self.engine.upscale(image)
