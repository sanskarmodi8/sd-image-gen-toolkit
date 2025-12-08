"""LORA Adapter loader module."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from sdgen.config import ASSETS_ROOT

# Assets/loras lives under src/assets/loras
LORA_DIR: Path = ASSETS_ROOT / "loras"
LORA_DIR.mkdir(parents=True, exist_ok=True)


def list_loras() -> List[str]:
    """Return a sorted list of available LoRA checkpoint filenames."""
    if not LORA_DIR.exists():
        return []
    return sorted([p.name for p in LORA_DIR.glob("*.safetensors")])


def get_lora_path(name: str) -> str:
    """Return the absolute path for a given LoRA filename."""
    return str(LORA_DIR / name)


def apply_loras(
    pipe,
    lora_a_name: Optional[str],
    alpha_a: float,
    lora_b_name: Optional[str],
    alpha_b: float,
) -> None:
    """Apply up to two LoRA adapters to the given pipeline.

    Uses diffusers' load_lora_weights / set_adapters API.

    Args:
        pipe: A Stable Diffusion pipeline instance.
        lora_a_name: Filename of first LoRA (or None).
        alpha_a: Weight for first LoRA.
        lora_b_name: Filename of second LoRA (or None).
        alpha_b: Weight for second LoRA.
    """
    # If the pipeline supports unloading adapters, clear previous ones
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    adapters = []
    weights = []

    if lora_a_name:
        pipe.load_lora_weights(
            get_lora_path(lora_a_name),
            adapter_name=Path(lora_a_name).stem,
        )
        adapters.append(Path(lora_a_name).stem)
        weights.append(float(alpha_a))

    if lora_b_name:
        pipe.load_lora_weights(
            get_lora_path(lora_b_name),
            adapter_name=Path(lora_b_name).stem,
        )
        adapters.append(Path(lora_b_name).stem)
        weights.append(float(alpha_b))

    if adapters and hasattr(pipe, "set_adapters"):
        pipe.set_adapters(adapters, weights)
