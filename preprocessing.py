"""Stateless image-processing utilities shared by all detectors."""

import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_contrast(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel to lift low-contrast text."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp-mask sharpening — makes text edges crisper before OCR."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def expand_roi(roi: dict, monitor: dict, pad: int) -> dict:
    """Return roi expanded by pad pixels in every direction, clamped to the monitor."""
    l = max(monitor["left"],                     roi["left"]  - pad)
    t = max(monitor["top"],                      roi["top"]   - pad)
    r = min(monitor["left"] + monitor["width"],  roi["left"]  + roi["width"]  + pad)
    b = min(monitor["top"]  + monitor["height"], roi["top"]   + roi["height"] + pad)
    return {"left": l, "top": t, "width": r - l, "height": b - t}
