"""UI for text to image generation section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import gradio as gr


@dataclass
class Txt2ImgControls:
    """UI element references for the Text → Image tab.

    These allow the Presets tab to populate the fields programmatically.
    """

    prompt: gr.components.Textbox
    negative: gr.components.Textbox
    steps: gr.components.Slider
    guidance: gr.components.Slider
    width: gr.components.Slider
    height: gr.components.Slider
    seed: gr.components.Textbox


def build_txt2img_tab(
    handler: Callable[..., Tuple],
    extra_inputs: Optional[List[gr.components.Component]] = None,
) -> Txt2ImgControls:
    """Construct the Text → Image tab and bind the Generate button."""
    with gr.Tab("Text → Image"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A futuristic city at dusk, cinematic lighting",
                )
                negative = gr.Textbox(
                    label="Negative prompt",
                    placeholder="low quality, blurry, extra limbs",
                )

                steps = gr.Slider(
                    minimum=10,
                    maximum=30,
                    value=20,
                    step=1,
                    label="Steps",
                )
                gr.Markdown(
                    "More steps → finer detail, slower runtime.",
                )

                guidance = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=7.5,
                    step=0.5,
                    label="Guidance Scale (CFG)",
                )
                gr.Markdown(
                    "Higher values make generation match \
the prompt more strictly. "
                )

                width = gr.Slider(
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                    label="Width",
                )
                height = gr.Slider(
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                    label="Height",
                )

                seed = gr.Textbox(
                    label="Seed (optional)",
                    value="",
                    placeholder="Leave empty for random",
                )

                generate_button = gr.Button("Generate")

            with gr.Column():
                out_image = gr.Image(label="Output")
                out_meta = gr.JSON(label="Metadata (JSON)")

        inputs = [prompt, negative, steps, guidance, width, height, seed]
        if extra_inputs:
            inputs.extend(extra_inputs)
        generate_button.click(
            fn=handler,
            inputs=inputs,
            outputs=[out_image, out_meta],
        )

    return Txt2ImgControls(
        prompt=prompt,
        negative=negative,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        seed=seed,
    )
