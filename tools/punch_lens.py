"""Interactive tool that makes the lens area of a glasses PNG semi-transparent.

Usage:
    glasses-punch-lens                 # file dialog
    glasses-punch-lens IMAGE_NOBG.png  # command line

Two trackbars let you tune the result per image:

* "Frame thickness (px)": estimated rim thickness in the source image.
  Adjust until the rim is not mistaken for lens (and vice versa).
* "Lens opacity (%)": how much of the original alpha the lens keeps.
  0 = fully transparent, 100 = unchanged. Tinted lenses usually look good
  around 35-55%.

The preview shows two panels: a checkerboard (to inspect alpha) and a rough
skin tone with two dark "eyes" (to check whether the eyes remain visible).

Keys: ENTER save, 'r' reset sliders, 'q' / ESC quit without saving.
The output is written next to the input with a ``_lenspunch`` suffix; the
original file is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..lens import DEFAULT_FRAME_THICKNESS, DEFAULT_LENS_ALPHA_PCT, punch_lens_transparency
from .common import ask_open_file, composite_over, make_checkerboard

TRACKBAR_THICKNESS = "Frame thickness (px)"
TRACKBAR_OPACITY = "Lens opacity (%)"
SKIN_TONE_BGR = (150, 180, 220)
KEY_ESC = 27


def make_skin_eye_backdrop(h: int, w: int) -> np.ndarray:
    """Create a crude skin-and-eyes backdrop to check lens see-through quality."""
    backdrop = np.full((h, w, 3), SKIN_TONE_BGR, dtype=np.uint8)
    eye_radius = max(4, w // 20)
    cy = h // 2
    for cx in (w // 3, w - w // 3):
        cv2.circle(backdrop, (cx, cy), eye_radius, (30, 30, 30), -1, lineType=cv2.LINE_AA)
    return backdrop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glasses-punch-lens",
        description="Make the lens area of a transparent glasses PNG semi-transparent.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="PNG with alpha (background removed).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    png_path = args.input or ask_open_file(
        "Select a glasses image with background removed (RGBA PNG)",
        [("PNG images", "*.png"), ("All files", "*.*")],
    )
    if not png_path or not png_path.exists():
        print("Error: file not found or selection cancelled.", file=sys.stderr)
        return 1

    rgba = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim < 3 or rgba.shape[2] != 4:
        print(
            "Error: the image has no alpha channel. Run glasses-remove-bg first.",
            file=sys.stderr,
        )
        return 1

    h, w = rgba.shape[:2]
    checker = make_checkerboard(h, w)
    skin_eyes = make_skin_eye_backdrop(h, w)

    window = f"Punch Lens - {png_path.name}"
    cv2.namedWindow(window)
    cv2.createTrackbar(TRACKBAR_THICKNESS, window, DEFAULT_FRAME_THICKNESS, 40, lambda v: None)
    cv2.createTrackbar(TRACKBAR_OPACITY, window, DEFAULT_LENS_ALPHA_PCT, 100, lambda v: None)
    print("Adjust the sliders. ENTER = save, 'r' = reset, 'q'/ESC = quit without saving.")

    result: Optional[np.ndarray] = None
    while True:
        thickness = cv2.getTrackbarPos(TRACKBAR_THICKNESS, window)
        opacity = cv2.getTrackbarPos(TRACKBAR_OPACITY, window)
        result, _ = punch_lens_transparency(rgba, thickness, opacity)

        preview = np.hstack([composite_over(result, checker), composite_over(result, skin_eyes)])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(preview, "Checkerboard", (10, 20), font, 0.5, (0, 0, 255), 1)
        cv2.putText(
            preview,
            "Skin + eyes (are the eyes still visible?)",
            (w + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.imshow(window, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), KEY_ESC):
            print("Quit without saving.")
            result = None
            break
        if key == ord("r"):
            cv2.setTrackbarPos(TRACKBAR_THICKNESS, window, DEFAULT_FRAME_THICKNESS)
            cv2.setTrackbarPos(TRACKBAR_OPACITY, window, DEFAULT_LENS_ALPHA_PCT)
        elif key in (13, 10):  # ENTER
            break

    cv2.destroyAllWindows()
    if result is None:
        return 0

    out_path = png_path.with_name(f"{png_path.stem}_lenspunch{png_path.suffix}")
    cv2.imwrite(str(out_path), result)

    opaque_before = int((rgba[:, :, 3] > 10).sum())
    opaque_after = int((result[:, :, 3] > 200).sum())
    print(f"Saved: {out_path}")
    print(f"Fully opaque pixels (frame): {opaque_before} -> {opaque_after}")
    print(f'Next step: glasses-pick-anchors "{out_path}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
