from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SAMWrapper:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        self.device = device
        self._sam = None
        self._predictor = None
        self._auto_gen = None
        self._model_path = Path(model_path) if model_path else _default_model_path()
        if self._model_path.exists():
            self._load()

    def _load(self):
        import torch
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_b"](checkpoint=str(self._model_path))
        sam.to(self.device)
        self._sam = sam
        self._predictor = SamPredictor(sam)
        self._auto_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=500,
        )

    @property
    def available(self) -> bool:
        return self._sam is not None

    def auto_segment(self, img: np.ndarray) -> list[dict[str, Any]]:
        """Return SAM auto masks sorted by area descending. img: float32 [0,1] RGB."""
        if not self.available:
            return []
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        masks = self._auto_gen.generate(u8)
        return sorted(masks, key=lambda m: m["area"], reverse=True)

    def prompt_segment(self, img: np.ndarray,
                       points: list[tuple[int, int]],
                       labels: list[int]) -> np.ndarray | None:
        """
        Run SAM with prompt points.
        points: list of (x, y) pixel coords
        labels: 1 = foreground, 0 = background
        Returns best mask as bool array (H, W).
        """
        if not self.available:
            return None
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        self._predictor.set_image(u8)
        pts = np.array(points, dtype=np.float32)
        lbs = np.array(labels, dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=pts,
            point_labels=lbs,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return masks[best]

    def threshold_fallback(self, img: np.ndarray,
                           threshold: float = 0.3) -> np.ndarray:
        """Simple luminance threshold when SAM is unavailable."""
        gray = img.mean(axis=2)
        return (gray > threshold).astype(np.uint8)


def _default_model_path() -> Path:
    return Path(__file__).parent.parent / "models" / "sam_vit_b.pth"


def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a bool mask as simple RLE for JSON storage."""
    flat = mask.flatten().tolist()
    rle, count = [], 0
    cur = flat[0] if flat else 0
    for v in flat:
        if v == cur:
            count += 1
        else:
            rle.append(count)
            count = 1
            cur = v
    rle.append(count)
    return {"size": list(mask.shape), "starts_with": int(flat[0]) if flat else 0, "counts": rle}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = []
    v = rle["starts_with"]
    for count in rle["counts"]:
        flat.extend([v] * count)
        v = 1 - v
    return np.array(flat, dtype=np.uint8).reshape(h, w).astype(bool)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    u8 = (mask.astype(np.uint8) * 255)
    _, buf = cv2.imencode(".png", u8)
    return buf.tobytes()


def png_bytes_to_mask(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img > 127
