"""Main entrypoint for the Stable Diffusion application.

This module initializes the text-to-image and image-to-image pipelines,
sets up the UI, and launches the Gradio interface.
"""

from __future__ import annotations

import os
import sys

# for HF spaces
sys.path.append(os.path.abspath("src"))

import torch
from dotenv import load_dotenv

from sdgen.config import AppSettings
from sdgen.sd.img2img import prepare_img2img_pipeline
from sdgen.sd.pipeline import load_pipeline, warmup_pipeline
from sdgen.ui import build_ui
from sdgen.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


def detect_device() -> str:
    """Return `"cuda"` if a GPU is available, otherwise `"cpu"`.

    Returns:
        The selected device string.
    """
    if torch.cuda.is_available():
        logger.info("CUDA available → using GPU")
        return "cuda"

    logger.warning("CUDA not detected → falling back to CPU")
    return "cpu"


def main() -> None:
    """Start the Stable Diffusion UI and initialize inference pipelines."""
    settings = AppSettings()
    model_id1 = settings.model_id1
    model_id2 = settings.model_id2

    device = "cpu"

    logger.info("Loading pipeline %s", model_id1)
    pipes = {
        "SD1.5": load_pipeline(
            model_id=model_id1,
            device=device,
            use_fp16=device == "cuda",
            enable_xformers=settings.enable_xformers,
        ),
        "Turbo": load_pipeline(
            model_id=model_id2,
            device=device,
            use_fp16=device == "cuda",
            enable_xformers=settings.enable_xformers,
        ),
    }
    if device == "cuda" and settings.warmup:
        warmup_pipeline(pipes["Turbo"])

    img2img_pipes = {
        "SD1.5": prepare_img2img_pipeline(pipes["SD1.5"]),
        "Turbo": prepare_img2img_pipeline(pipes["Turbo"]),
    }

    demo = build_ui(pipes, img2img_pipes)
    demo.launch(
        server_name=settings.server_host,
        server_port=settings.server_port,
        share=settings.share,
    )


if __name__ == "__main__":
    main()
