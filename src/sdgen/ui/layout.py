"""UI layout builder for the Stable Diffusion Gradio app."""

from __future__ import annotations

from typing import Any, Tuple

import gradio as gr

from sdgen.sd.generator import generate_image
from sdgen.sd.img2img import generate_img2img
from sdgen.sd.models import Img2ImgConfig, Txt2ImgConfig
from sdgen.ui.tabs import (
    build_history_tab,
    build_img2img_tab,
    build_presets_tab,
    build_txt2img_tab,
    build_upscaler_tab,
)
from sdgen.upscaler.upscaler import Upscaler
from sdgen.utils.common import pretty_json, to_pil
from sdgen.utils.history import save_history_entry
from sdgen.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_seed(value: Any) -> int | None:
    """Return integer seed if valid, otherwise None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        logger.warning("Invalid seed input: %s", value)
        return None


def _update_steps_and_cfg(model):
    """Upate steps based on the model."""
    if model == "Turbo":
        return (
            gr.update(minimum=1, maximum=5, value=2, step=1),
            gr.update(minimum=0, maximum=0, value=0, step=0),
        )
    return (
        gr.update(minimum=10, maximum=30, value=20, step=1),
        gr.update(minimum=1, maximum=10, value=5, step=1),
    )


def _txt2img_handler(
    model_choice: str,
    pipes: dict,
    prompt: str,
    negative: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Any,
) -> Tuple[Any, str]:
    """Run text-to-image generation."""
    model = model_choice
    pipe = pipes[model]

    cfg = Txt2ImgConfig(
        prompt=prompt or "",
        negative_prompt=negative or "",
        steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        seed=_resolve_seed(seed),
        device=pipe.device.type,
    )

    image, meta = generate_image(pipe, cfg)

    try:
        save_history_entry(meta, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return image, pretty_json(meta.to_dict())


def _img2img_handler(
    model_choice: str,
    pipes: dict,
    input_image: Any,
    prompt: str,
    negative: str,
    strength: float,
    steps: int,
    guidance: float,
    seed: Any,
) -> Tuple[Any, str]:
    """Run image-to-image generation."""
    model = model_choice
    pipe = pipes[model]

    if input_image is None:
        raise gr.Error("Upload an image to continue.")

    pil_image = to_pil(input_image)

    cfg = Img2ImgConfig(
        prompt=prompt or "",
        negative_prompt=negative or "",
        strength=float(strength),
        steps=int(steps),
        guidance_scale=float(guidance),
        width=pil_image.width,
        height=pil_image.height,
        seed=_resolve_seed(seed),
        device=pipe.device.type,
    )

    image, meta = generate_img2img(pipe, cfg, pil_image)

    try:
        save_history_entry(meta, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return image, pretty_json(meta.to_dict())


def _upscale_handler(
    input_image: Any,
    scale: str,
) -> Tuple[Any, str]:
    """Run image upscaling."""
    if input_image is None:
        raise gr.Error("Upload an image to continue.")

    pil_image = to_pil(input_image)

    # scale is str → convert to int
    try:
        scale_int = int(float(scale))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error("Scale must be numeric (2 or 4).") from exc

    upscaler = Upscaler(scale=scale_int, prefer="ncnn")
    out_image, meta = upscaler.upscale(pil_image)

    try:
        save_history_entry(meta, out_image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return out_image, pretty_json(meta.to_dict())


def make_img2img_handler(model_choice, pipes):
    """Return handler for img2img generation."""

    def handler(input_image, prompt, negative, strength, steps, guidance, seed):
        return _img2img_handler(
            model_choice,
            pipes,
            input_image,
            prompt,
            negative,
            strength,
            steps,
            guidance,
            seed,
        )

    return handler


def make_txt2img_handler(model_choice, pipes):
    """Return handler for txt2img generation."""

    def handler(prompt, negative, steps, guidance, width, height, seed):
        return _txt2img_handler(
            model_choice,
            pipes,
            prompt,
            negative,
            steps,
            guidance,
            width,
            height,
            seed,
        )

    return handler


def build_ui(txt2img_pipes: dict, img2img_pipes: dict) -> gr.Blocks:
    """Build the entire Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Stable Diffusion Generator\n"
            "Generation can take from few seconds to several minutes based \
on the Model selceted and applied settings."
        )

        model_choice = gr.Dropdown(
            choices=[
                "SD1.5",
                "Turbo",
            ],
            value="SD1.5",
            label="Model",
        )
        gr.Markdown(
            "Use Turbo model if you prefer speed over quality. \
SD1.5 produces much better results but takes 10 minutes on average \
for generation on HF Spaces."
        )

        txt_controls = build_txt2img_tab(
            make_txt2img_handler(model_choice.value, txt2img_pipes),
        )

        img_controls = build_img2img_tab(
            make_img2img_handler(model_choice.value, img2img_pipes),
        )

        build_upscaler_tab(
            handler=_upscale_handler,
        )

        build_presets_tab(
            txt_controls=txt_controls,
            img_controls=img_controls,
        )

        build_history_tab()

        model_choice.change(
            fn=_update_steps_and_cfg,
            inputs=[model_choice],
            outputs=[txt_controls.steps, txt_controls.guidance],
        )
        model_choice.change(
            fn=_update_steps_and_cfg,
            inputs=[model_choice],
            outputs=[img_controls.steps, img_controls.guidance],
        )

        gr.Markdown(
            "### Notes\n"
            "- Use **History → Refresh** if new entries do not appear.\n"
            "- Presets apply to both **Text → Image** and \
**Image → Image** tabs.\n"
            "- This app is hosted on CPU based HF Spaces, so generation may \
take some time, please be patient.\n"
        )

    return demo
