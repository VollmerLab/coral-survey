"""
WhiBal G7 detection and QR code detection.
Card dimensions come from config — never hardcoded here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class CardDetection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    scale_mm_px: float | None
    confidence: float


@dataclass
class QRDetection:
    genotype_id: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def detect_whibal(
    img: np.ndarray,
    card_size_mm: tuple[float, float] = (53.3, 85.1),
    corner: str = "lower-left",
    search_fraction: float = 0.35,
) -> CardDetection | None:
    """
    Detect WhiBal G7 card in img (float32 [0,1] RGB).

    Strategy:
      1. Search in expected corner (position prior).
      2. Fall back to full-frame quad detection.

    Returns CardDetection or None.
    card_size_mm: (width, height) in mm. From config — don't hardcode.
    """
    h, w = img.shape[:2]
    result = _corner_search(img, h, w, corner, search_fraction, card_size_mm)
    if result is None:
        result = _fullframe_quad(img, card_size_mm)
    return result


def _corner_search(img, h, w, corner, frac, card_size_mm):
    if corner == "lower-left":
        roi = img[int(h * (1 - frac)):, :int(w * frac)]
        oy, ox = int(h * (1 - frac)), 0
    elif corner == "upper-left":
        roi = img[:int(h * frac), :int(w * frac)]
        oy, ox = 0, 0
    elif corner == "lower-right":
        roi = img[int(h * (1 - frac)):, int(w * (1 - frac)):]
        oy, ox = int(h * (1 - frac)), int(w * (1 - frac))
    else:
        roi = img[:int(h * frac), int(w * (1 - frac)):]
        oy, ox = 0, int(w * (1 - frac))

    det = _detect_gray_rect(roi, card_size_mm)
    if det is None:
        return None
    bx, by, bw, bh = det.bbox
    det.bbox = (bx + ox, by + oy, bw, bh)
    det.contour = det.contour + np.array([ox, oy])
    return det


def _fullframe_quad(img, card_size_mm):
    return _detect_gray_rect(img, card_size_mm)


def _detect_gray_rect(img: np.ndarray, card_size_mm: tuple[float, float]) -> CardDetection | None:
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    img_area = h * w

    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.005 or area > img_area * 0.4:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 0
        expected_aspect = max(card_size_mm) / min(card_size_mm)
        if abs(aspect - expected_aspect) > 0.4:
            continue

        gray_roi = gray[y:y+bh, x:x+bw]
        mean_gray = float(gray_roi.mean())
        std_gray = float(gray_roi.std())

        # WhiBal is a neutral gray card: medium luminance, low std
        if not (80 < mean_gray < 180):
            continue
        if std_gray > 40:
            continue

        score = 1.0 / (1.0 + abs(mean_gray - 128) / 50.0 + std_gray / 20.0)
        if score > best_score:
            best_score = score
            px_per_mm = max(bw, bh) / max(card_size_mm)
            best = CardDetection(
                bbox=(x, y, bw, bh),
                contour=approx.reshape(-1, 2),
                scale_mm_px=1.0 / px_per_mm if px_per_mm > 0 else None,
                confidence=score,
            )

    return best


def detect_qr(img: np.ndarray) -> QRDetection | None:
    """Detect QR code and decode genotype ID. img: float32 [0,1] RGB."""
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(gray)
    if data and points is not None:
        pts = points[0].astype(int)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        w = pts[:, 0].max() - x
        h = pts[:, 1].max() - y
        return QRDetection(genotype_id=data.strip(), bbox=(x, y, w, h))
    return None


def sample_card_face(img: np.ndarray, bbox: tuple[int, int, int, int],
                     qr_bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """
    Return pixels from the card face, excluding the QR code region.
    img: float32 [0,1] RGB. Returns 2D array of pixels (N, 3).
    """
    x, y, w, h = bbox
    roi = img[y:y+h, x:x+w]
    mask = np.ones((h, w), dtype=bool)

    if qr_bbox is not None:
        qx, qy, qw, qh = qr_bbox
        qx_rel = qx - x
        qy_rel = qy - y
        margin = 5
        y1 = max(0, qy_rel - margin)
        y2 = min(h, qy_rel + qh + margin)
        x1 = max(0, qx_rel - margin)
        x2 = min(w, qx_rel + qw + margin)
        mask[y1:y2, x1:x2] = False

    pixels = roi[mask]
    return pixels
