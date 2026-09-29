"""A locally loaded image model with single-image and batch prediction.

Default: torchvision ResNet-18 trained on ImageNet (weights download once, ~45 MB).
To serve your own model, keep the ``predict_one`` / ``predict_batch`` interface and
change ``_load`` (and ``labels``) - the server does not need to change.

This class is NOT safe to call from several threads at once. The server makes
sure only one call runs at a time.
"""
import io
import logging
from typing import Dict, List

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

log = logging.getLogger("serving.local_model")


class ImageClassifier:
    def __init__(self, device: str = "cpu", top_k: int = 5):
        self.device = torch.device(device)
        self.top_k = top_k
        self._load()

    def _load(self) -> None:
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        self.labels: List[str] = weights.meta["categories"]
        log.info("Loaded ResNet-18 on %s", self.device)

    @staticmethod
    def decode(image_bytes: bytes) -> Image.Image:
        """Bytes -> RGB image. Raises ValueError for anything that isn't a readable image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as e:
            raise ValueError("Not a valid image file") from e
        return image.convert("RGB")

    @torch.inference_mode()
    def predict_batch(self, images: List[Image.Image]) -> List[List[Dict[str, float]]]:
        """One forward pass for the whole batch - much faster per image than looping."""
        if not images:
            return []
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)
        top_probs, top_ids = probs.topk(self.top_k, dim=1)
        return [
            [{"label": self.labels[i], "score": round(p, 4)} for p, i in zip(ps.tolist(), ids.tolist())]
            for ps, ids in zip(top_probs, top_ids)
        ]

    def predict_one(self, image: Image.Image) -> List[Dict[str, float]]:
        return self.predict_batch([image])[0]

    def warmup(self) -> None:
        """Run one dummy prediction so the first real request isn't slow."""
        self.predict_one(Image.new("RGB", (224, 224)))
