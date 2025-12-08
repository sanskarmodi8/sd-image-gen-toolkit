"""Configuration dataclasses for Stable Diffusion execution and history storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Txt2ImgConfig:
    """Configuration for text-to-image generation.

    Attributes:
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        steps: Number of diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        width: Requested image width.
        height: Requested image height.
        seed: Optional random seed.
        device: Target torch device ("cuda" or "cpu").
    """

    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    device: str = "cuda"


@dataclass
class Img2ImgConfig:
    """Configuration for image-to-image generation.

    Attributes:
        prompt: Positive prompt text.
        init_image_path: Optional file path to source image.
        negative_prompt: Negative prompt text.
        strength: Img2Img blend strength in (0, 1].
        steps: Number of diffusion steps.
        guidance_scale: CFG scale.
        width: Requested image width.
        height: Requested image height.
        seed: Optional random seed.
        device: Target device.
    """

    prompt: str
    init_image_path: Optional[str] = None
    negative_prompt: str = ""
    strength: float = 0.7
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    device: str = "cuda"


@dataclass
class GenerationMetadata:
    """Generic metadata for any generation mode.

    Fields are optional depending on mode:
    - Txt2Img: prompt, negative, steps, guidance
    - Img2Img: prompt, negative, strength, steps, guidance
    - Upscale: scale, original size, final size
    """

    mode: str  # "txt2img", "img2img", "upscale"

    # Shared
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    id: Optional[str] = None
    thumbnail: Optional[str] = None
    full_image: Optional[str] = None
    model_id: Optional[str] = None

    # Txt2Img / Img2Img
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    lora_names: Optional[List[str]] = None
    lora_alphas: Optional[List[float]] = None

    # Img2Img only
    strength: Optional[float] = None

    # Upscale only
    scale: Optional[float] = None
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Drop None values for clean JSON."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HistorySummary:
    """Minimal entry used for UI history lists."""

    id: str
    prompt: str
    mode: str
    seed: Optional[int]
    width: int
    height: int
    timestamp: str
    thumbnail: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation."""
        return asdict(self)
