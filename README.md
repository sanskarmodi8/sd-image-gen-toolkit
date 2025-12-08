---
title: sd-image-gen-toolkit
app_file: src/sdgen/main.py
sdk: gradio
sdk_version: 6.0.2
---

# Stable Diffusion Image Generation Toolkit

![appdemo](https://drive.google.com/uc?export=view&id=1dO2bnYmEEj3fNU0-dV692icUPSwyP93G)

[**Live Demo**](https://huggingface.co/spaces/SanskarModi/sd-image-gen-toolkit)

---

## Overview

A modular, lightweight image generation toolkit built on **Hugging Face Diffusers**, designed for **CPU-friendly deployment**, clean architecture, and practical usability.

It supports **Text → Image**, **Image → Image**, and **Upscaling**, with a **preset system**, optional **LoRA adapters**, and a local **metadata history** for reproducibility.

---

## Features

### Text → Image
- Stable Diffusion **1.5** and **Turbo**
- Configurable prompt parameters:
  - prompt / negative prompt
  - steps
  - guidance (CFG)
  - resolution
  - seed (optional)
- JSON metadata output
- Style presets for quick experimentation

### Image → Image
- Modify existing images via the SD Img2Img pipeline
- Denoising strength control
- Full parameter configuration
- Shared preset system
- History saved for reproducibility

### Upscaling (Real-ESRGAN NCNN)
- **2× and 4×** upscaling
- NCNN backend (no GPU required)
- Minimal dependencies
- Fast on CPU environments (HF Spaces)

### LoRA Adapter Support
- Runtime loading of `.safetensors` adapters
- Up to **two adapters** with independent weights
- Alpha range `-2 → +2` per adapter
- Automatic discovery under:
```

src/assets/loras/

```
- LoRA UI is **disabled for Turbo**, since Turbo does not benefit from LoRA injection

### Metadata History
Every generation stores:
- model id
- prompt + negative prompt
- steps, cfg, resolution
- seed
- LoRA names + weights
- timestamp

All generated data is stored in a tree structure under:
```

src/assets/history/

```

---

## Architecture

```

src/
└── sdgen/
├── sd/                     # Stable Diffusion runtime
│   ├── pipeline.py         # model loading, device config
│   ├── generator.py        # text-to-image inference
│   ├── img2img.py          # image-to-image inference
│   ├── lora_loader.py      # LoRA discovery & injection
│   └── models.py           # typed config & metadata objects
│
├── ui/                     # Gradio UI components
│   ├── layout.py           # composition root for UI
│   └── tabs/               # modular tabs
│       ├── txt2img_tab.py
│       ├── img2img_tab.py
│       ├── upscaler_tab.py
│       ├── presets_tab.py
│       └── history_tab.py
│
├── presets/                # curated basic presets
│   └── styles.py           # preset registry
│
├── upscaler/               # Real-ESRGAN NCNN backend
│   ├── upscaler.py         # interface + metadata
│   └── realesrgan.py       # NCNN wrapper
│
├── utils/                  # shared utilities
│   ├── history.py          # atomic storage format
│   ├── common.py           # PIL helpers
│   └── logger.py           # structured logging
│
└── config/                 # static configuration
├── paths.py            # resolved directories
└── settings.py         # environment settings

````

---

## Presets (Included)

The project includes **four style presets**, each defining:

- prompt
- negative prompt

These presets are neutral and work with both **SD1.5** and **Turbo**:

| Name               | Style                     |
|--------------------|----------------------------|
| Realistic Photo    | 35mm, photorealistic       |
| Anime              | clean anime illustration   |
| Cinematic / Moody  | cinematic lighting/grain   |
| Oil Painting       | classical oil painting     |

Presets do **not include LoRA parameters**.  
Users may manually combine presets with LoRA adapters.

---

## Installation

### Clone
```bash
git clone https://github.com/sanskarmodi8/stable-diffusion-image-generator
cd stable-diffusion-image-generator
````

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies (CPU)

```bash
pip install -r requirements.txt
pip install -e .
```

### GPU (optional)

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

---

## Run

```bash
python src/sdgen/main.py
```

Open in browser:

```
http://127.0.0.1:7860
```

---

## Adding LoRA Models

Place `.safetensors` files here:

```
src/assets/loras/
```

They will be automatically detected and displayed in the UI (SD1.5 only).

This repository **does not include** LoRA files.

---

## Third-Party LoRA Models

The app supports optional LoRA adapters.
LoRA weights are **not included** and are **the property of their respective authors**.

If you choose to download LoRA files automatically (see `lora_urls.py`), they are fetched directly from their original sources (**Civitai**).

This project does **not** redistribute LoRA weights.
Refer to each model’s license on Civitai.

---

## Development

The repo uses `pre-commit` hooks for consistency:

```bash
pre-commit install
```

Tools:

* ruff
* black
* isort

Check formatting:

```bash
ruff check .
black .
```

---

## License

This project is licensed under the **MIT License**.
See the [`LICENSE`](LICENSE) file.

---

## Author 

[**Sanskar Modi**](https://github.com/sanskarmodi8)