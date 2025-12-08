---
title: sd-image-gen-toolkit
app_file: src/sdgen/main.py
sdk: gradio
sdk_version: 6.0.2
---

# Stable Diffusion Image Generator Toolkit

![appdemo](https://drive.google.com/uc?export=view&id=1B2AJT_MB-3U9Uw9f-fzhJmVHZYgPBuCm)

A modular image generation system built on **HuggingFace Diffusers**, with support for multiple Stable Diffusion pipelines, configurable inference parameters, a clean **Gradio UI**, and a lightweight local **history/metadata store**.

The system supports **text-to-image**, **image-to-image**, and **super-resolution upscaling** using **Real-ESRGAN (NCNN)**.
Designed with a focus on **extensibility**, **clean code**, and **practical deployment constraints** (CPU or low-memory environments).

[Visit App](https://huggingface.co/spaces/SanskarModi/sd-image-gen-toolkit)

---

# Core Features

## Text-to-Image Generation

* Stable Diffusion pipelines (SD 1.5, Turbo)
* Adjustable **CFG scale**, **inference steps**, resolution, and seed
* Structured metadata (JSON) for reproducibility
* Style presets with recommended parameters

## Image-to-Image (Img2Img)

* Pipeline reuse to avoid model reload cost
* Alpha-preserving prompt transforms
* Configurable denoising strength
* Deterministic or stochastic sampling

## Upscaling (Real-ESRGAN NCNN)

* Lightweight **NCNN backend** (GPU not required)
* Supports 2× and 4× scaling
* Optional SD-upscaler backend planned
* Minimal dependencies, fast on CPU

## Prompt History & Metadata Tracking

* Local metadata index with atomic writes
* Thumbnail + full-size image storage
* JSON schema for portability
* History browser UI

## Multi-Model Runtime Switching

* Multiple pipelines loaded once
* Selection at inference without reload
* Shared tokenizer/encoder where possible
* Warm-up logic for fast Turbo inference

---

# Architecture Overview

```
src/sdgen/
│
├── sd/
│   ├── pipeline.py          # pipeline loader, warmup, dtype/device logic
│   ├── generator.py         # text-to-image
│   ├── img2img.py           # image-to-image
│   └── models.py            # config/metadata dataclasses
│
├── ui/
│   ├── layout.py            # top-level UI composition
│   └── tabs/                # individual UI components
│
├── presets/
│   └── styles.py            # curated style presets
│
├── upscaler/
│   └── realesrgan.py        # NCNN Real-ESRGAN backend
│
├── utils/
│   ├── history.py           # persistence layer
│   ├── common.py            # PIL/NumPy helpers
│   └── logger.py            # structured logging
│
└── config/
    ├── settings.py          # runtime config/env
    └── paths.py             # project paths
```

---

# Technical Highlights

### Efficient CPU Deployment

HF Spaces have **no GPU**, 16 GB RAM.
Generation speed is optimized via:

* latent consistency (Turbo)
* reduced step ranges
* VAE tiling for memory distribution
* attention slicing
* deferring safety checker if private

This reduces **CPU inference from ~220s → <70s** for 512px prompts, without unacceptable quality loss.

### Multi-Pipeline Switching

Both SD pipelines are instantiated once.
The UI passes `model_choice` to the handler, which selects the correct pipeline **without rebuilding**.

This avoids 4-7 GB reload cost per click.

---

# Local Installation

### 1. Clone

```bash
git clone https://github.com/sanskarmodi8/stable-diffusion-image-generator
cd stable-diffusion-image-generator
```

### 2. Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install PyTorch for GPU (leave if on CPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install core libs:

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. HuggingFace Login (optional)

```bash
huggingface-cli login
```

---

# Running

```bash
python src/sdgen/main.py
```

UI available at:

```
http://127.0.0.1:7860
```

---

# Roadmap (Focused, High-Impact Features)

This project is under active development. The next milestones focus on **practical model customization and multi-model support**, optimized for **CPU-only deployment environments** such as Hugging Face Spaces.

The roadmap is intentionally **lean** to maximize value within limited compute constraints.

---

## 1. LoRA Runtime Inference (Core Feature)

Add lightweight **Low-Rank Adaptation** support for Stable Diffusion pipelines without modifying base model weights.

### Scope
- Load external **`.safetensors` LoRA adapters** into UNet
- Apply LoRA modules dynamically at inference
- **Alpha (weight) slider** to control influence
- **UI dropdown** for selecting LoRA adapters
- **Automatic discovery** of LoRAs under:
```

src/assets/loras/

```

### Deliverables
- `lora_loader.py` utility
- integration into existing `load_pipeline()`
- UI: LoRA selector + alpha parameter
- history metadata with:
- `lora_paths`
- `lora_weights`

---

## 2. Multi-LoRA Mixing (2 adapters)

Support mixing **two LoRA adapters** with independent weights.

### Scope
- Simple weighted merge at attention processors
- UI:
- LoRA A dropdown + alpha
- LoRA B dropdown + alpha
- Conflict handling for overlapping layers

### Deliverables
- `apply_lora_mix()` utility
- metadata persistence

---

## 3. SDXL-Turbo Pipeline Support

Add a **third runtime model**:
```

stabilityai/stable-diffusion-xl-base
stabilityai/sdxl-turbo

````

### Scope
- instantiate SDXL Turbo pipeline
- auto configure:
  - steps (1-4)
  - CFG (0-1)
- model selection integrated in UI
- reproducible metadata

### Notes
SDXL Turbo is optimized for **fast generation** and works well on constrained environments with reduced steps.

---

## 4. Enhanced Presets

Presets currently define only prompts. Extend them to define **full recommended parameter sets** per use case.

### Scope
Each preset can define:
- prompt
- negative prompt
- inference steps
- CFG scale
- resolution
- recommended model
- recommended LoRA (+alpha)

### Example
```json
{
  "preset": "Anime Portrait",
  "prompt": "...",
  "negative": "...",
  "steps": 15,
  "cfg": 6,
  "width": 512,
  "height": 768,
  "model": "SD1.5",
  "lora": {
    "path": "anime_face.safetensors",
    "alpha": 0.8
  }
}
````

---

## 5. Metadata Improvements

Enhance metadata tracking for **reproducibility**.

### Added Fields

* `model_id`
* `lora_names`
* `lora_alphas`
* `preset_used`
* `resolution`
* provenance timestamp

This enables exact replication of generated images.

---

## 6. Example LoRA & Training Scripts (No UI)

Provide **self-contained example** to demonstrate training:

* a Colab notebook for **LoRA fine-tuning**
* a small 20-image dataset
* training duration < 45 minutes on free GPU
* export `.safetensors` file
* use it in presets

### Deliverables

* `examples/train_lora.ipynb`
* resulting LoRA stored at `assets/loras/example.safetensors`

---

# Contributing

This repo is configured with **pre-commit**:

* black
* ruff
* isort
* docstring linting (Google style)

Install hooks:

```bash
pre-commit install
```

Test formatting:

```bash
ruff check .
black .
```

Branching convention:

```
feat/<feature>
fix/<issue>
refactor/<module>
```

---

# License

This project is licensed under [MIT License](LICENSE).

---

# Author

**Sanskar Modi**

Machine Learning Engineer
Focused on production-grade ML systems.

GitHub: [https://github.com/sanskarmodi8](https://github.com/sanskarmodi8)