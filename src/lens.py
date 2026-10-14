"""Lens transparency processing (color-independent, shape-based).

Earlier approaches separated "frame" from "lens" by brightness, which only
works for clear or light lenses. Dark lenses (e.g. sunglasses) are as dark
as the frame, so they were never made transparent and hid the wearer's eyes.

This module separates them by *shape* instead:

* The frame is always a thin band running around the lens (the rim).
* The lens is always a thick, solid region inside that rim.

Eroding the object mask by roughly the rim thickness removes the thin rim
and leaves only the lens "core", regardless of lens color. The frame keeps
its original alpha; the lens alpha is scaled down so the wearer's eyes show
through a tinted, semi-transparent lens.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

DEFAULT_FRAME_THICKNESS = 10  # px, estimated rim thickness
DEFAULT_LENS_ALPHA_PCT = 40  # %, opacity kept inside the lens

MIN_OBJECT_ALPHA = 30  # pixels with alpha above this belong to the glasses
CLOSE_KERNEL = 5  # closes gaps in the rim caused by highlights/reflections
CLOSE_ITERATIONS = 2


def compute_lens_core_mask(object_mask: np.ndarray, frame_thickness: int) -> np.ndarray:
    """Return a mask of the thick lens regions, excluding the thin frame rim."""
    frame_thickness = max(1, int(frame_thickness))
    size = frame_thickness * 2 + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    lens_core = cv2.erode(object_mask, erode_kernel)

    # Grow back slightly for a smoother region, but never beyond the object.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lens_core = cv2.dilate(lens_core, dilate_kernel)
    return cv2.bitwise_and(lens_core, object_mask)


def punch_lens_transparency(
    rgba: np.ndarray,
    frame_thickness: int,
    lens_alpha_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make the lens area semi-transparent while keeping the frame opaque.

    Args:
        rgba: Input BGRA image (background already removed).
        frame_thickness: Estimated rim thickness in pixels.
        lens_alpha_pct: Percentage (0-100) of the original alpha kept inside
            the lens. 0 = fully transparent, 100 = unchanged.

    Returns:
        ``(result_bgra, lens_core_mask)``.
    """
    alpha = rgba[:, :, 3]
    object_mask = (alpha > MIN_OBJECT_ALPHA).astype(np.uint8) * 255

    close_kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    object_mask = cv2.morphologyEx(
        object_mask, cv2.MORPH_CLOSE, close_kernel, iterations=CLOSE_ITERATIONS
    )

    lens_core = compute_lens_core_mask(object_mask, frame_thickness)

    factor = float(np.clip(lens_alpha_pct, 0, 100)) / 100.0
    result = rgba.copy()
    new_alpha = alpha.copy()
    lens = lens_core > 0
    new_alpha[lens] = np.clip(alpha[lens].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    result[:, :, 3] = new_alpha
    return result, lens_core
