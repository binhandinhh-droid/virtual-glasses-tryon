"""Loading of glasses assets (RGBA image + anchor metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from .config import ANCHOR_KEYS

PathLike = Union[str, Path]

_FALLBACK_FRAME_COLOR_BGR = (20, 20, 20)


class GlassesAsset:
    """A single glasses model.

    Holds a BGRA image together with four anchor points stored in a sidecar
    JSON file (same base name as the PNG):

    * ``left_eye_center`` / ``right_eye_center``: lens centers, used to fit
      the glasses to the wearer's eyes.
    * ``left_temple_anchor`` / ``right_temple_anchor``: hinge points where
      the temple arms start, used to draw the arms dynamically.
    """

    def __init__(self, png_path: PathLike):
        png_path = Path(png_path)
        json_path = png_path.with_suffix(".json")

        image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {png_path}")
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Glasses image must have an alpha channel (RGBA): {png_path}")
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing anchor file: {json_path}. Run `glasses-pick-anchors` on the image first."
            )

        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        missing = [k for k in ANCHOR_KEYS if k not in meta]
        if missing:
            raise ValueError(f"Anchor file {json_path} is missing keys: {missing}")

        self.image: np.ndarray = image  # BGRA
        self.name: str = png_path.name
        self.left_anchor = np.array(meta["left_eye_center"], dtype=np.float32)
        self.right_anchor = np.array(meta["right_eye_center"], dtype=np.float32)
        self.left_temple_anchor = np.array(meta["left_temple_anchor"], dtype=np.float32)
        self.right_temple_anchor = np.array(meta["right_temple_anchor"], dtype=np.float32)

        # Sampled at the hinges so the dynamically drawn temples match the frame.
        self.frame_color: Tuple[int, int, int] = self._sample_frame_color()

    @property
    def eye_distance(self) -> float:
        """Distance between the two lens centers in asset pixel space."""
        return float(np.linalg.norm(self.right_anchor - self.left_anchor))

    def _sample_frame_color(self) -> Tuple[int, int, int]:
        """Average the color of opaque pixels around both hinge anchors."""
        h, w = self.image.shape[:2]
        samples = []
        for anchor in (self.left_temple_anchor, self.right_temple_anchor):
            x = int(np.clip(round(float(anchor[0])), 0, w - 1))
            y = int(np.clip(round(float(anchor[1])), 0, h - 1))
            for dx in range(-6, 7):
                xx = int(np.clip(x + dx, 0, w - 1))
                if self.image[y, xx, 3] > 100:
                    samples.append(self.image[y, xx, :3])
        if samples:
            return tuple(int(c) for c in np.mean(samples, axis=0))
        return _FALLBACK_FRAME_COLOR_BGR


def load_all_glasses(asset_dir: PathLike) -> List[GlassesAsset]:
    """Load every ``*.png`` in ``asset_dir`` that has a matching ``.json`` file."""
    asset_dir = Path(asset_dir)
    pngs = sorted(asset_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(
            f"No glasses images found in '{asset_dir}'. "
            "Prepare assets with `glasses-remove-bg`, `glasses-punch-lens` and "
            "`glasses-pick-anchors` first."
        )
    return [GlassesAsset(p) for p in pngs]
