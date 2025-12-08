"""UI layout builder for the Stable Diffusion Gradio app."""

from __future__ import annotations

from typing import Any, Tuple

import gradio as gr

from sdgen.sd import (
    Img2ImgConfig,
    Txt2ImgConfig,
    apply_loras,
    generate_image,
    generate_img2img,
    list_loras,
)
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


# Small helpers


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
        logger.warning("Invalid seed input for seed: %s", value)
        return None


def _normalize_lora_name(raw: Any) -> str | None:
    """Normalize dropdown value to a LoRA filename or None."""
    if raw is None:
        return None
    name = str(raw).strip()
    if not name or name == "(none)":
        return None
    return name


def _steps_cfg_for_model(model: str) -> Tuple[gr.update, gr.update]:
    """Return UI updates for steps & CFG when the model changes."""
    if model == "Turbo":
        # Turbo: super low steps, CFG has no meaningful effect.
        return (
            gr.update(minimum=1, maximum=5, value=2, step=1),
            gr.update(minimum=0, maximum=0, value=0, step=0),
        )

    # SD1.5 defaults
    return (
        gr.update(minimum=10, maximum=30, value=20, step=1),
        gr.update(minimum=1, maximum=10, value=5, step=1),
    )


def _validate_turbo_strength(
    model: str,
    steps: float,
    strength: float,
) -> Tuple[gr.update, gr.update, str]:
    """UI constraint: Turbo requires steps * strength >= 1."""
    if model != "Turbo":
        return gr.update(), gr.update(), ""

    product = float(steps) * float(strength)
    if product >= 1.0:
        return gr.update(), gr.update(), f"Turbo OK: {product:.2f} ≥ 1"

    required = 1.0 / float(steps)
    required = min(required, 1.0)
    return (
        gr.update(),
        gr.update(value=required),
        "Adjusted for Turbo: steps×strength must be ≥ 1",
    )


def _update_model_dependents(model: str):
    """update.

    - steps & cfg sliders for txt2img and img2img
    - LoRA panel visibility
    """
    steps_t2i, cfg_t2i = _steps_cfg_for_model(model)
    steps_i2i, cfg_i2i = _steps_cfg_for_model(model)
    lora_visibility = gr.update(visible=(model == "SD1.5"))

    return (
        steps_t2i,
        cfg_t2i,
        steps_i2i,
        cfg_i2i,
        lora_visibility,
    )


def _apply_lora_if_allowed(
    model: str,
    pipe: Any,
    lora_a: str | None,
    alpha_a: float,
    lora_b: str | None,
    alpha_b: float,
) -> Tuple[list[str], list[float]]:
    """Apply up to two LoRA adapters to the given pipeline.

    - Only applied for SD1.5.
    - Turbo completely ignores LoRA (and unloads any existing ones).
    - Returns (active_lora_names, active_lora_alphas) for metadata.
    """
    if model != "SD1.5":
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to unload LoRA weights: %s", exc)
        return [], []

    names: list[str] = []
    alphas: list[float] = []

    # Filter out zero-weight adapters to keep metadata clean.
    if lora_a and alpha_a != 0:
        names.append(lora_a)
        alphas.append(float(alpha_a))

    if lora_b and alpha_b != 0:
        names.append(lora_b)
        alphas.append(float(alpha_b))

    if not names:
        # Nothing to apply
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to unload LoRA weights: %s", exc)
        return [], []

    apply_loras(
        pipe,
        names[0] if len(names) > 0 else None,
        alphas[0] if len(alphas) > 0 else 0.0,
        names[1] if len(names) > 1 else None,
        alphas[1] if len(alphas) > 1 else 0.0,
    )

    return names, alphas


# Core handlers


def _txt2img_handler(
    model: str,
    pipes: dict[str, Any],
    prompt: str,
    negative: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Any,
    lora_a_value: Any,
    lora_a_alpha_value: Any,
    lora_b_value: Any,
    lora_b_alpha_value: Any,
) -> Tuple[Any, str]:
    """Run text-to-image generation."""
    pipe = pipes[model]

    lora_a = _normalize_lora_name(lora_a_value)
    lora_b = _normalize_lora_name(lora_b_value)
    alpha_a = float(lora_a_alpha_value or 0.0)
    alpha_b = float(lora_b_alpha_value or 0.0)

    active_lora_names, active_lora_alphas = _apply_lora_if_allowed(
        model,
        pipe,
        lora_a,
        alpha_a,
        lora_b,
        alpha_b,
    )

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
    meta.model_id = model
    meta.lora_names = active_lora_names
    meta.lora_alphas = active_lora_alphas

    try:
        save_history_entry(meta, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return image, pretty_json(meta.to_dict())


def _img2img_handler(
    model: str,
    pipes: dict[str, Any],
    input_image: Any,
    prompt: str,
    negative: str,
    strength: float,
    steps: int,
    guidance: float,
    seed: Any,
    lora_a_value: Any,
    lora_a_alpha_value: Any,
    lora_b_value: Any,
    lora_b_alpha_value: Any,
) -> Tuple[Any, str]:
    """Run image-to-image generation."""
    pipe = pipes[model]

    if input_image is None:
        raise gr.Error("Upload an image to continue.")

    pil_image = to_pil(input_image)

    lora_a = _normalize_lora_name(lora_a_value)
    lora_b = _normalize_lora_name(lora_b_value)
    alpha_a = float(lora_a_alpha_value or 0.0)
    alpha_b = float(lora_b_alpha_value or 0.0)

    active_lora_names, active_lora_alphas = _apply_lora_if_allowed(
        model,
        pipe,
        lora_a,
        alpha_a,
        lora_b,
        alpha_b,
    )

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
    meta.model_id = model
    meta.lora_names = active_lora_names
    meta.lora_alphas = active_lora_alphas

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

    try:
        scale_int = int(float(scale))
    except Exception as exc:  # noqa: BLE001
        raise gr.Error("Scale must be numeric (2 or 4).") from exc

    upscaler = Upscaler(scale=scale_int, prefer="ncnn")
    out_image, meta = upscaler.upscale(pil_image)
    meta.model_id = "RealESRGAN"

    try:
        save_history_entry(meta, out_image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save history entry: %s", exc)

    return out_image, pretty_json(meta.to_dict())


# Handler factories wired to Gradio


def make_txt2img_handler(pipes: dict[str, Any]):
    """Factory to build the txt2img handler with extra UI inputs."""

    def handler(
        prompt: str,
        negative: str,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        seed: Any,
        model_choice_value: Any,
        lora_a_value: Any,
        lora_a_alpha_value: Any,
        lora_b_value: Any,
        lora_b_alpha_value: Any,
    ):
        model = str(model_choice_value)
        return _txt2img_handler(
            model,
            pipes,
            prompt,
            negative,
            steps,
            guidance,
            width,
            height,
            seed,
            lora_a_value,
            lora_a_alpha_value,
            lora_b_value,
            lora_b_alpha_value,
        )

    return handler


def make_img2img_handler(pipes: dict[str, Any]):
    """Factory to build the img2img handler with extra UI inputs."""

    def handler(
        input_image: Any,
        prompt: str,
        negative: str,
        strength: float,
        steps: int,
        guidance: float,
        seed: Any,
        model_choice_value: Any,
        lora_a_value: Any,
        lora_a_alpha_value: Any,
        lora_b_value: Any,
        lora_b_alpha_value: Any,
    ):
        model = str(model_choice_value)
        return _img2img_handler(
            model,
            pipes,
            input_image,
            prompt,
            negative,
            strength,
            steps,
            guidance,
            seed,
            lora_a_value,
            lora_a_alpha_value,
            lora_b_value,
            lora_b_alpha_value,
        )

    return handler


# Top-level UI composition


def build_ui(txt2img_pipes: dict, img2img_pipes: dict) -> gr.Blocks:
    """Build the entire Gradio UI."""
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Stable Diffusion Generator\n"
            "SD1.5 - slower, higher quality | Turbo - faster, lower quality"
        )

        model_choice = gr.Dropdown(
            choices=["SD1.5", "Turbo"],
            value="SD1.5",
            label="Model",
        )

        # LoRA controls (SD1.5 only). Hidden when Turbo is selected.
        with gr.Accordion("LoRA", open=False) as lora_group:
            lora_files = list_loras()
            lora_choices = ["(none)"] + lora_files

            lora_a = gr.Dropdown(
                lora_choices,
                value="(none)",
                label="LoRA A",
                info=(
                    "Primary LoRA. Use this for style/character/detail control. "
                    "Pick '(none)' to disable."
                ),
            )
            alpha_a = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="LoRA A weight",
                info=(
                    "Positive → apply effect. Negative → \
dampen or invert. 0 → disable."
                ),
            )

            lora_b = gr.Dropdown(
                lora_choices,
                value="(none)",
                label="LoRA B (optional)",
                info="Optional second LoRA. Can be mixed with LoRA A.",
            )
            alpha_b = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="LoRA B weight",
                info=("Same convention as LoRA A. Use lighter weights when mixing."),
            )

        # Core tabs
        txt_controls = build_txt2img_tab(
            make_txt2img_handler(txt2img_pipes),
            extra_inputs=[model_choice, lora_a, alpha_a, lora_b, alpha_b],
        )

        img_controls = build_img2img_tab(
            make_img2img_handler(img2img_pipes),
            extra_inputs=[model_choice, lora_a, alpha_a, lora_b, alpha_b],
        )

        build_upscaler_tab(handler=_upscale_handler)
        build_presets_tab(
            txt_controls=txt_controls,
            img_controls=img_controls,
            model_choice=model_choice,
            lora_a=lora_a,
            alpha_a=alpha_a,
            lora_b=lora_b,
            alpha_b=alpha_b,
        )
        build_history_tab()

        model_choice.change(
            fn=_update_model_dependents,
            inputs=[model_choice],
            outputs=[
                txt_controls.steps,
                txt_controls.guidance,
                img_controls.steps,
                img_controls.guidance,
                lora_group,
            ],
        )

        # turbo constraints
        msg = gr.Markdown("", visible=False)
        for inp in [model_choice, img_controls.steps, img_controls.strength]:
            inp.change(
                fn=_validate_turbo_strength,
                inputs=[model_choice, img_controls.steps, img_controls.strength],
                outputs=[img_controls.steps, img_controls.strength, msg],
            )

        gr.Markdown(
            "### Notes\n"
            "- **History → Refresh** if new entries do not appear.\n"
            "- Presets apply to both **Text → Image** and **Image → Image** tabs.\n"
            "- Deployed on CPU-only HF Spaces, so performance would be a little slower \
(~10 mins for SD1.5 and ~1.5 min for Turbo on default settings)."
        )

    return demo
