"""Interactive tool for picking anchor points on a glasses PNG.

Usage:
    glasses-pick-anchors                 # file dialog
    glasses-pick-anchors IMAGE.png       # command line

The PNG must have a transparent background (alpha channel).

Click order (4 points):
    1. Center of the LEFT lens  (left side of the image)
    2. Center of the RIGHT lens
    3. LEFT hinge  (outer edge of the frame where the left temple starts)
    4. RIGHT hinge

Keys:
    u        undo the last point
    r        reset all points
    ENTER    save the JSON file (after all 4 points are placed)
    q / ESC  quit without saving

The JSON file is written next to the PNG with the same base name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

from ..config import ANCHOR_KEYS
from .common import ask_open_file, composite_over, make_checkerboard

PROMPTS = (
    "1. LEFT lens center (middle of the lens on the left side of the image)",
    "2. RIGHT lens center (middle of the lens on the right side of the image)",
    "3. LEFT hinge (outer frame edge where the left temple starts)",
    "4. RIGHT hinge (outer frame edge where the right temple starts)",
)

# Maximum size of the display window; larger images are scaled down for display.
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 800
KEY_ESC = 27


def display_scale(width: int, height: int) -> float:
    """Scale factor (<= 1) that fits an image of the given size on screen."""
    return min(MAX_DISPLAY_W / width, MAX_DISPLAY_H / height, 1.0)


def to_original_coords(points: Sequence[Tuple[int, int]], scale: float) -> List[Tuple[int, int]]:
    """Convert clicked display coordinates back to original image coordinates."""
    return [(round(x / scale), round(y / scale)) for x, y in points]


def build_anchor_dict(points: Sequence[Tuple[int, int]], image_size: Tuple[int, int]) -> dict:
    """Build the JSON-serializable anchor metadata."""
    anchors = {key: list(point) for key, point in zip(ANCHOR_KEYS, points)}
    anchors["image_size"] = list(image_size)
    return anchors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glasses-pick-anchors",
        description="Pick lens-center and hinge anchor points on a glasses PNG.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="PNG with a transparent background.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    png_path = args.input or ask_open_file(
        "Select a glasses image (PNG with transparent background)",
        [("PNG images", "*.png"), ("All files", "*.*")],
    )
    if not png_path or not png_path.exists():
        print("Error: image not found or selection cancelled.", file=sys.stderr)
        return 1

    print(f"Opening image: {png_path}")
    rgba = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        print(f"Error: could not read image: {png_path}", file=sys.stderr)
        return 1
    if rgba.ndim < 3 or rgba.shape[2] != 4:
        print(
            "Error: the image has no alpha channel (transparent background).\n"
            "Remove the background first with glasses-remove-bg.",
            file=sys.stderr,
        )
        return 1

    h, w = rgba.shape[:2]
    base_full = composite_over(rgba, make_checkerboard(h, w))

    scale = display_scale(w, h)
    if scale < 1.0:
        disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
        base_display = cv2.resize(base_full, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        print(
            f"Image {w}x{h}px is larger than the screen limit; displayed at {disp_w}x{disp_h}px "
            f"(scale {scale:.3f}). Clicks are mapped back to the original image automatically."
        )
    else:
        base_display = base_full

    points: List[Tuple[int, int]] = []  # in display coordinates
    window = f"Pick Anchors - {png_path.name}"

    def redraw() -> None:
        img = base_display.copy()
        for i, p in enumerate(points):
            cv2.circle(img, p, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                img, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        if len(points) < len(PROMPTS):
            message = PROMPTS[len(points)]
        else:
            message = "All 4 points set. ENTER = save, 'u' = undo, 'r' = reset."
        cv2.putText(img, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        cv2.imshow(window, img)

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(PROMPTS):
            points.append((x, y))
            redraw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    print("Click the 4 points in order, as described in the image window.")

    exit_code = 0
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), KEY_ESC):
            print("Quit without saving.")
            break
        if key == ord("u") and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key in (13, 10) and len(points) == len(PROMPTS):
            anchors = build_anchor_dict(to_original_coords(points, scale), (w, h))
            json_path = png_path.with_suffix(".json")
            json_path.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
            print(f"Saved: {json_path}")
            print(json.dumps(anchors, indent=2))
            break

    cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
