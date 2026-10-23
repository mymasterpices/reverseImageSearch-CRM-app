# crmApp/image-service/embedder.py
"""
Image/text embedding wrapper.

EMBED_MODEL options (set in .env):
  google/siglip2-base-patch16-224    default, great quality/speed balance, supports text search
  google/siglip2-so400m-patch14-384  best quality, ~4x slower (GPU recommended)
  facebook/dinov2-base               best for pure image->image "looks the same" matching, no text search
  openai/clip-vit-base-patch32       the old model
"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoImageProcessor

MODEL_NAME = os.getenv("EMBED_MODEL", "google/siglip2-base-patch16-224")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_tensor(out) -> torch.Tensor:
    """transformers versions differ: tensor vs output object."""
    if isinstance(out, torch.Tensor):
        return out
    for key in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, key, None)
        if v is not None:
            return v
    return out.last_hidden_state[:, 0]


class Embedder:
    def __init__(self, name: str = MODEL_NAME):
        self.name = name
        self.device = pick_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.is_dino = "dino" in name.lower()
        self.supports_text = not self.is_dino

        self.model = AutoModel.from_pretrained(name, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = (AutoImageProcessor if self.is_dino else AutoProcessor).from_pretrained(name)
        self.dim = self.encode_images([Image.new("RGB", (64, 64))]).shape[1]

    @staticmethod
    def _norm(x: torch.Tensor) -> np.ndarray:
        x = x.float()
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_images(self, images: list) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  for k, v in inputs.items()}
        if self.is_dino:
            out = self.model(pixel_values=inputs["pixel_values"])
        else:
            out = self.model.get_image_features(**inputs)
        return self._norm(_as_tensor(out))

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if not self.supports_text:
            raise ValueError(f"{self.name} does not support text queries")
        inputs = self.processor(text=[text.lower()], padding="max_length",
                                max_length=64, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._norm(_as_tensor(self.model.get_text_features(**inputs)))[0]
