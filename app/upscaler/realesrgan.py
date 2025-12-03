"""NCNN RealESRGAN upscaler wrapper.

This module exposes:
- NCNNUpscaler: provides lightweight 2x/4x super-resolution via realesrgan-ncnn-py.
"""

from __future__ import annotations

from PIL import Image
from realesrgan_ncnn_py import Realesrgan

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Supported scales mapped to internal model indices
SCALE_TO_MODEL = {
    2.0: 3,  # realesrgan-x2plus
    4.0: 0,  # realesrgan-x4plus
}


class NCNNUpscaler:
    """Lightweight NCNN RealESRGAN engine using realesrgan-ncnn-py.

    Args:
        scale (float): Supported values = 2.0 or 4.0.
    """

    def __init__(self, scale: float = 2.0):
        """Initialize the NCNN upscaler."""
        if scale not in SCALE_TO_MODEL:
            raise ValueError("Only 2.0x and 4.0x supported for your NCNN build")

        self.scale = scale
        self.model_index = SCALE_TO_MODEL[scale]

        logger.info(
            f"[NCNN] Loading RealESRGAN model index={self.model_index} \
            for scale={scale}x"
        )

        self.model = Realesrgan(model=self.model_index)

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale a PIL image using NCNN RealESRGAN."""
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image")

        logger.info(
            f"[NCNN] Upscaling ({image.width}x{image.height}) "
            f"by {self.scale}x using model={self.model_index}"
        )

        return self.model.process_pil(image)
