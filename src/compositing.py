"""Alpha compositing helpers."""

from __future__ import annotations

import numpy as np


def alpha_over(top_bgra: np.ndarray, bottom_bgra: np.ndarray) -> np.ndarray:
    """Composite two BGRA layers using the Porter-Duff "over" operator.

    ``top_bgra`` is placed over ``bottom_bgra``. Both must have the same shape.
    """
    top_a = top_bgra[:, :, 3:4].astype(np.float32) / 255.0
    bot_a = bottom_bgra[:, :, 3:4].astype(np.float32) / 255.0
    top_rgb = top_bgra[:, :, :3].astype(np.float32)
    bot_rgb = bottom_bgra[:, :, :3].astype(np.float32)

    out_a = top_a + bot_a * (1.0 - top_a)
    out_rgb = top_rgb * top_a + bot_rgb * bot_a * (1.0 - top_a)
    out_rgb = out_rgb / np.clip(out_a, 1e-6, 1.0)  # un-premultiply; avoid divide-by-zero

    out = np.zeros_like(top_bgra, dtype=np.uint8)
    out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return out


def overlay_rgba(frame_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """Blend a BGRA overlay onto an opaque BGR frame and return a BGR image."""
    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = overlay_bgra[:, :, :3].astype(np.float32)
    bg = frame_bgr.astype(np.float32)
    return (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
