import cv2
import numpy as np


WHIBAL_G7_L_REF = 49.5  # L* for 18% gray under D65


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """img: float32 [0,1] RGB → float32 Lab."""
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    # OpenCV encodes Lab as L*[0,255], a*+128, b*+128
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def sample_region(img: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Return mean L*, a*, b* within mask (bool or 0/1 uint8)."""
    lab = rgb_to_lab(img)
    m = mask.astype(bool)
    if not m.any():
        return 0.0, 0.0, 0.0
    return (
        float(lab[:, :, 0][m].mean()),
        float(lab[:, :, 1][m].mean()),
        float(lab[:, :, 2][m].mean()),
    )


def whibal_correction_factor(img: np.ndarray, card_mask: np.ndarray) -> float:
    """Correction factor = ref L* / measured L*. Multiply image L* by this."""
    l_measured, _, _ = sample_region(img, card_mask)
    if l_measured <= 0:
        return 1.0
    return WHIBAL_G7_L_REF / l_measured


def apply_lab_correction(lab: np.ndarray, factor: float) -> np.ndarray:
    corrected = lab.copy()
    corrected[:, :, 0] *= factor
    return corrected


def area_cm2(area_px: int, scale_mm_px: float) -> float:
    area_mm2 = area_px * (scale_mm_px ** 2)
    return area_mm2 / 100.0
