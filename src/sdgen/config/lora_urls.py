"""URLs for external LoRA adapters.

These URLs point directly to the original creators' downloads on Civitai.
We download the adapters at runtime if they are not present locally.

Legal note:
We do NOT redistribute LoRA weights in this repository.
Users download them from the original source at runtime.
"""

from __future__ import annotations

from typing import Dict

# Direct API download endpoints from Civitai.
# we expect to have under `assets/loras/`.
LORA_URLS: Dict[str, str] = {
    "DetailTweak.safetensors": "https://civitai.com/api/download/models/62833?type=Model&format=SafeTensor",
    "MangaPanels.safetensors": "https://civitai.com/api/download/models/28907?type=Model&format=SafeTensor&size=full&fp=fp16",
    "AnimeTarotCards.safetensors": "https://civitai.com/api/download/models/28609?type=Model&format=SafeTensor&size=full&fp=fp16",
}
