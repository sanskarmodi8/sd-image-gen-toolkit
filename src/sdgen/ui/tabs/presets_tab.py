"""UI for presets section."""

from __future__ import annotations

from typing import Any, Tuple

import gradio as gr

from sdgen.presets.styles import get_preset, list_presets
from sdgen.ui.tabs.img2img_tab import Img2ImgControls
from sdgen.ui.tabs.txt2img_tab import Txt2ImgControls


def apply_preset(
    preset_name: Any,
    model_name: Any,
) -> Tuple[Any, ...]:
    """Return values for txt2img and img2img controls based on the chosen preset.

    Args:
        preset_name: A string or a one-element list representing the preset key.
        model_name: The current model choice, used to disable LoRA when Turbo.

    Returns:
        Tuple of UI values in a fixed field order.
    """
    if isinstance(preset_name, (list, tuple)):
        preset_name = preset_name[0] if preset_name else None

    if not preset_name:
        raise gr.Error("Select a preset first.")

    preset = get_preset(str(preset_name))
    if preset is None:
        raise gr.Error("Invalid preset selected.")

    model = str(model_name).strip()

    prompt = preset.get("prompt", "")
    negative = preset.get("negative_prompt", "")
    steps = int(preset.get("steps", 20))
    cfg = float(preset.get("cfg", 5.0))
    width = int(preset.get("width", 512))
    height = int(preset.get("height", 512))
    loraA = preset.get("lora_A")
    alphaA = preset.get("alpha_A", 0.0)
    loraB = preset.get("lora_B")
    alphaB = preset.get("alpha_B", 0.0)

    # Turbo mode → ignore LoRA and override sampler settings
    if model == "Turbo":
        steps = 2
        cfg = 0.0
        loraA = "(none)"
        alphaA = 0.0
        loraB = "(none)"
        alphaB = 0.0
    else:
        # normalize empty
        loraA = loraA or "(none)"
        loraB = loraB or "(none)"

    status_msg = f"Applied preset: {preset_name}"

    return (
        # txt2img
        prompt,
        negative,
        steps,
        cfg,
        width,
        height,
        loraA,
        alphaA,
        loraB,
        alphaB,
        # img2img
        prompt,
        negative,
        steps,
        cfg,
        # status
        status_msg,
    )


def build_presets_tab(
    txt_controls: Txt2ImgControls,
    img_controls: Img2ImgControls,
    model_choice: gr.Dropdown,
    lora_a: gr.Dropdown,
    alpha_a: gr.Slider,
    lora_b: gr.Dropdown,
    alpha_b: gr.Slider,
) -> None:
    """Create the Presets tab and connect values to UI fields."""
    with gr.Tab("Presets"):
        with gr.Row():
            with gr.Column():
                preset_name = gr.Dropdown(
                    choices=list_presets(),
                    label="Select style",
                )
                apply_button = gr.Button("Apply Preset")
                status_box = gr.Markdown("")

        apply_button.click(
            fn=apply_preset,
            inputs=[preset_name, model_choice],
            outputs=[
                # txt2img
                txt_controls.prompt,
                txt_controls.negative,
                txt_controls.steps,
                txt_controls.guidance,
                txt_controls.width,
                txt_controls.height,
                lora_a,
                alpha_a,
                lora_b,
                alpha_b,
                # img2img
                img_controls.prompt,
                img_controls.negative,
                img_controls.steps,
                img_controls.guidance,
                # status box
                status_box,
            ],
        )
