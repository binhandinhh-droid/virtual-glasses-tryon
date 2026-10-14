"""Geometric transforms that fit a glasses asset onto a detected face."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from .asset import GlassesAsset


def _third_point(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return a point perpendicular to segment p1->p2 (half its length)."""
    d = p2 - p1
    perp = np.array([-d[1], d[0]], dtype=np.float32)
    return p1 + perp * 0.5


def compute_affine(
    glasses: GlassesAsset,
    face_left_eye: Sequence[float],
    face_right_eye: Sequence[float],
) -> np.ndarray:
    """Compute the 2x3 similarity transform mapping asset space to frame space.

    The two lens-center anchors are mapped onto the detected eye positions.
    ``cv2.getAffineTransform`` needs three point pairs, so a virtual third
    point is added perpendicular to the eye line in both spaces. This yields
    a transform with uniform scale, rotation (head roll) and translation.

    The same matrix is used to warp the glasses image and to project the
    hinge anchors into frame coordinates.
    """
    src = np.array([glasses.left_anchor, glasses.right_anchor], dtype=np.float32)
    dst = np.array([face_left_eye, face_right_eye], dtype=np.float32)

    src3 = np.vstack([src, _third_point(src[0], src[1])])
    dst3 = np.vstack([dst, _third_point(dst[0], dst[1])])
    return cv2.getAffineTransform(src3, dst3)


def transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """Apply a 2x3 affine matrix to a 2D point."""
    homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
    return (matrix @ homogeneous)[:2]


def warp_glasses_image(
    glasses: GlassesAsset,
    matrix: np.ndarray,
    frame_shape: Tuple[int, ...],
) -> np.ndarray:
    """Warp the glasses image into frame coordinates.

    Returns a BGRA image with the same width/height as the frame.
    """
    h, w = frame_shape[:2]
    return cv2.warpAffine(
        glasses.image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
