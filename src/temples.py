"""Dynamic rendering of the glasses temple arms."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from .asset import GlassesAsset
from .config import (
    TEMPLE_BASE_THICKNESS_PX,
    TEMPLE_CURVATURE,
    TEMPLE_CURVE_SEGMENTS,
    TEMPLE_TAPER,
)
from .geometry import transform_point


def draw_dynamic_temples(
    frame_shape: Tuple[int, ...],
    glasses: GlassesAsset,
    matrix: np.ndarray,
    left_ear_pt: Sequence[float],
    right_ear_pt: Sequence[float],
    scale_ratio: float,
) -> np.ndarray:
    """Draw both temple arms on a transparent BGRA layer.

    Each arm runs from the (transformed) hinge anchor of the frame to the
    face landmark near the ear, so the arms follow the wearer's face width
    and head tilt instead of being a fixed line baked into the still image.

    Args:
        frame_shape: Shape of the camera frame; the layer matches its size.
        glasses: The glasses asset (provides hinge anchors and frame color).
        matrix: Asset-to-frame affine transform from ``compute_affine``.
        left_ear_pt: Frame-space target point for the left arm.
        right_ear_pt: Frame-space target point for the right arm.
        scale_ratio: Current face scale relative to the asset; scales thickness.
    """
    h, w = frame_shape[:2]
    layer = np.zeros((h, w, 4), dtype=np.uint8)

    left_hinge = transform_point(matrix, glasses.left_temple_anchor)
    right_hinge = transform_point(matrix, glasses.right_temple_anchor)
    thickness = max(2, int(round(TEMPLE_BASE_THICKNESS_PX * scale_ratio)))

    for hinge, ear in ((left_hinge, left_ear_pt), (right_hinge, right_ear_pt)):
        _draw_temple_curve(layer, hinge, ear, glasses.frame_color, thickness)
    return layer


def _draw_temple_curve(
    layer: np.ndarray,
    p_hinge: Sequence[float],
    p_ear: Sequence[float],
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw one slightly curved arm (quadratic Bezier) that tapers toward the ear."""
    p1 = np.array(p_hinge, dtype=np.float32)
    p2 = np.array(p_ear, dtype=np.float32)
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length < 1:
        return

    perp = np.array([-d[1], d[0]], dtype=np.float32) / length
    # Control point bows the arm slightly to mimic wrapping around the cheek.
    ctrl = p1 + d * 0.5 + perp * (length * TEMPLE_CURVATURE)

    n = TEMPLE_CURVE_SEGMENTS
    points = []
    for i in range(n + 1):
        t = i / n
        points.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * ctrl + t**2 * p2)

    for i in range(n):
        t = i / n
        segment_thickness = max(1, int(round(thickness * (1 - TEMPLE_TAPER * t))))
        p_a = tuple(int(v) for v in np.round(points[i]))
        p_b = tuple(int(v) for v in np.round(points[i + 1]))
        cv2.line(layer, p_a, p_b, (*color_bgr, 255), segment_thickness, lineType=cv2.LINE_AA)
