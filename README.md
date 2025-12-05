---
title: sd-image-gen-toolkit
app_file: src/sdgen/main.py
sdk: gradio
sdk_version: 6.0.2
---

# Stable Diffusion Image Generator

A modular image generation system built on **HuggingFace Diffusers**, with support for multiple Stable Diffusion pipelines, configurable inference parameters, a clean **Gradio UI**, and a lightweight local **history/metadata store**.

The system supports **text-to-image**, **image-to-image**, and **super-resolution upscaling** using **Real-ESRGAN (NCNN)**.
Designed with a focus on **extensibility**, **clean code**, and **practical deployment constraints** (CPU or low-memory environments).

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

# Roadmap (LoRA, QLoRA, and Training)

**Update planned**: full LoRA loading and fine-tuning support.

Scope includes:

### 1. LoRA Runtime Inference

* Load LoRA weights into existing UNet
* Adjustable LoRA alpha/scaling
* UI selector for LoRA checkpoints
* Enable mixing multiple LoRAs

Implementation plan:

* Attach `lora_attn_procs` to model
* Discover `.safetensors` in `/assets/lora`
* Store LoRA metadata in history
* Persist alpha value and presets

### 2. QLoRA Fine-Tuning

* Train lightweight LoRA modules on GPUs (11GB VRAM OK)
* Use parameter-efficient training
* Merge adapters for export
* Allow user fine-tuning via command line

Stack:

* accelerate
* peft
* bitsandbytes (if GPU available)

UI tab planned:

* dataset upload
* config builder
* start training
* track loss, sample outputs

**Why LoRA?**

* Enables personal styles without training the full model
* Reduces VRAM and compute cost by 50–200×
* Industry-standard for SD customization

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