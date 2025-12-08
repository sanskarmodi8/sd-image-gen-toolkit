"""Preset configurations."""

from __future__ import annotations

from typing import Any, Dict, List

PRESETS: Dict[str, Dict[str, Any]] = {
    "Cinematic Realism": {
        "prompt": (
            "ultra realistic, cinematic lighting, 35mm film look, depth "
            "of field, sharp focus, natural skin texture"
        ),
        "negative_prompt": (
            "lowres, blurry, deformed anatomy, \
extra limbs, oversaturated, jpeg artifacts"
        ),
        "steps": 24,
        "cfg": 6.5,
        "width": 768,
        "height": 512,
        "lora_A": "DetailTweak.safetensors",
        "alpha_A": 0.9,
        "lora_B": None,
        "alpha_B": 0.0,
    },
    "Oil Painting / Classic Art": {
        "prompt": (
            "oil painting, impasto brush strokes, classical \ \
lighting, Rembrandt style"
        ),
        "negative_prompt": "blurry, cartoonish, digital artifacts",
        "steps": 20,
        "cfg": 7.5,
        "width": 512,
        "height": 512,
        "lora_A": "DetailTweak.safetensors",
        "alpha_A": 0.8,
        "lora_B": None,
        "alpha_B": 0.0,
    },
    "Manga Illustration": {
        "prompt": (
            "manga illustration, clean line art, expressive pose, full "
            "background, detailed composition"
        ),
        "negative_prompt": "badhandsv4, easyn, blurry line art",
        "steps": 20,
        "cfg": 7.0,
        "width": 512,
        "height": 704,
        "lora_A": "MangaPanels.safetensors",
        "alpha_A": 1.0,
        "lora_B": None,
        "alpha_B": 0.0,
    },
    "Anime Tarot": {
        "prompt": (
            "anime tarot card, ornate composition, symbolic character pose, "
            "intricate patterns, layered design"
        ),
        "negative_prompt": "badhandsv4, flat background, simple layout",
        "steps": 20,
        "cfg": 6.0,
        "width": 512,
        "height": 704,
        "lora_A": "AnimeTarotCards.safetensors",
        "alpha_A": 1.0,
        "lora_B": "DetailTweak.safetensors",
        "alpha_B": -0.4,
    },
}


def get_preset(name: str) -> Dict[str, Any] | None:
    """Return shallow copy of a preset config."""
    data = PRESETS.get(name)
    return dict(data) if data else None


def list_presets() -> List[str]:
    """Stable UI order."""
    return list(PRESETS.keys())
