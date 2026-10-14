"""Realism effects applied around the glasses."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import SHADOW_BLUR_KSIZE, SHADOW_OFFSET, SHADOW_STRENGTH


def apply_shadow(
    frame_bgr: np.ndarray,
    glasses_bgra: np.ndarray,
    offset: Tuple[int, int] = SHADOW_OFFSET,
    strength: float = SHADOW_STRENGTH,
) -> np.ndarray:
    """Darken the skin beneath the glasses to fake a soft drop shadow.

    The glasses alpha channel is shifted by ``offset`` (dx, dy), blurred and
    used as a multiplicative darkening mask on the frame.
    """
    alpha = glasses_bgra[:, :, 3].astype(np.float32) / 255.0
    mask = np.roll(alpha, shift=offset[1], axis=0)
    mask = np.roll(mask, shift=offset[0], axis=1)
    mask = cv2.GaussianBlur(mask, (SHADOW_BLUR_KSIZE, SHADOW_BLUR_KSIZE), 0) * strength
    mask3 = cv2.merge([mask] * 3)
    return (frame_bgr.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)
