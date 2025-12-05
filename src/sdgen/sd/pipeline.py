"""Stable Diffusion pipeline loading and warmup helpers."""

from __future__ import annotations

import os
from typing import Optional

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
)

from sdgen.utils.logger import get_logger

logger = get_logger(__name__)


def _try_enable_xformers(pipe: StableDiffusionPipeline) -> None:
    """Enable xFormers memory-efficient attention if available."""
    try:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory-efficient attention.")
    except Exception as exc:
        logger.info("xFormers not enabled: %s", exc)


def load_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    use_fp16: bool = True,
    enable_xformers: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    scheduler: any = None,
) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline with optional scheduler and xFormers.

    Args:
        model_id: HuggingFace model ID.
        device: Execution device ("cuda" or "cpu").
        use_fp16: Enable float16 precision on CUDA.
        enable_xformers: Whether to enable xFormers attention.
        torch_dtype: Explicit dtype override.
        scheduler: Optional preconfigured scheduler.

    Returns:
        A configured `StableDiffusionPipeline` instance.
    """
    if torch_dtype is None:
        torch_dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32

    if scheduler is None:
        try:
            if "turbo" in model_id.lower():
                scheduler = LCMScheduler.from_pretrained(
                    model_id,
                    subfolder="scheduler",
                )
            else:
                scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    model_id,
                    subfolder="scheduler",
                )
        except Exception:
            scheduler = None

    logger.info(
        "Loading pipeline %s dtype=%s on %s",
        model_id,
        torch_dtype,
        device,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        scheduler=scheduler,
        use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    ).to(device)

    try:
        pipe.enable_attention_slicing()
        logger.info("Enabled attention slicing.")
    except Exception:
        logger.info("Attention slicing not available.")

    if enable_xformers:
        _try_enable_xformers(pipe)

    try:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
            logger.info("Enabled VAE tiling.")
    except Exception:
        pass

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("Pipeline loaded.")
    return pipe


def warmup_pipeline(
    pipe: StableDiffusionPipeline,
    prompt: str = "A photo of a cat",
    height: int = 512,
    width: int = 512,
) -> None:
    """Run a one-step warmup pass to initialize CUDA kernels."""
    try:
        if hasattr(pipe, "parameters"):
            device = next(pipe.parameters()).device
        else:
            device = "cuda"
    except Exception:
        device = "cuda"

    try:
        gen_device = "cpu" if str(device) == "cpu" else device
        generator = torch.Generator(gen_device).manual_seed(0)

        logger.info("Warmup: running one-step inference to initialize kernels.")
        pipe(
            prompt=prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=height,
            width=width,
            generator=generator,
        )

        if device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Warmup complete.")
    except Exception as exc:
        logger.warning("Warmup failed: %s", exc)
