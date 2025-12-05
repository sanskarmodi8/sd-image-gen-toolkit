"""UI for image to image generation section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import gradio as gr


@dataclass
class Img2ImgControls:
    """References to Image → Image controls used by the presets tab."""

    input_image: gr.Image
    prompt: gr.Textbox
    negative: gr.Textbox
    strength: gr.Slider
    steps: gr.Slider
    guidance: gr.Slider
    seed: gr.Textbox


def build_img2img_tab(handler: Callable[..., Tuple[Any, dict]]) -> Img2ImgControls:
    """Build the Image → Image tab and connect it to the provided handler.

    Args:
        handler: A callable accepting the UI inputs and returning:
            (output_image, metadata_dict)

    Returns:
        Img2ImgControls: A container with references to UI components.
    """
    with gr.Tab("Image → Image"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe desired changes...",
                )

                negative = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Artifacts to avoid...",
                )

                strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Strength",
                )
                gr.Markdown(
                    "Controls how strongly the prompt \
                        alters the original image."
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
                    value=7.0,
                    step=0.5,
                    label="Guidance Scale",
                )
                gr.Markdown(
                    "Higher values make generation match \
the prompt more strictly. "
                )

                seed = gr.Textbox(
                    label="Seed",
                    value="",
                    placeholder="Leave empty for random",
                )
                gr.Markdown("If using Turbo model, ensure steps * strength >= 1")
                generate_button = gr.Button("Generate")

            with gr.Column():
                out_image = gr.Image(
                    label="Output",
                    type="pil",
                )
                out_metadata = gr.JSON(
                    label="Metadata",
                )

        generate_button.click(
            fn=handler,
            inputs=[
                input_image,
                prompt,
                negative,
                strength,
                steps,
                guidance,
                seed,
            ],
            outputs=[out_image, out_metadata],
        )

    return Img2ImgControls(
        input_image=input_image,
        prompt=prompt,
        negative=negative,
        strength=strength,
        steps=steps,
        guidance=guidance,
        seed=seed,
    )
