"""Real-time webcam try-on application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .asset import GlassesAsset, load_all_glasses
from .compositing import alpha_over, overlay_rgba
from .config import (
    DEFAULT_ASSET_DIR,
    LEFT_EYE_OUTER_IDX,
    LEFT_TEMPLE_IDX,
    RIGHT_EYE_OUTER_IDX,
    RIGHT_TEMPLE_IDX,
)
from .effects import apply_shadow
from .geometry import compute_affine, warp_glasses_image
from .temples import draw_dynamic_temples

WINDOW_TITLE = "Virtual Eyeglass Try-On"
KEY_ESC = 27


def render_glasses(
    frame: np.ndarray,
    landmarks,
    glasses: GlassesAsset,
    show_shadow: bool,
) -> np.ndarray:
    """Render ``glasses`` onto ``frame`` using the given Face Mesh landmarks."""
    h, w = frame.shape[:2]

    def point(idx: int) -> np.ndarray:
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

    left_eye = point(LEFT_EYE_OUTER_IDX)
    right_eye = point(RIGHT_EYE_OUTER_IDX)
    left_temple = point(LEFT_TEMPLE_IDX)
    right_temple = point(RIGHT_TEMPLE_IDX)

    matrix = compute_affine(glasses, left_eye, right_eye)
    front = warp_glasses_image(glasses, matrix, frame.shape)

    # Face scale relative to the asset drives the temple thickness.
    face_eye_distance = float(np.linalg.norm(right_eye - left_eye))
    scale_ratio = face_eye_distance / max(glasses.eye_distance, 1e-6)

    temples = draw_dynamic_temples(
        frame.shape, glasses, matrix, left_temple, right_temple, scale_ratio
    )
    # Frame and lenses go on top of the temple arms.
    glasses_layer = alpha_over(front, temples)

    output = apply_shadow(frame, glasses_layer) if show_shadow else frame
    return overlay_rgba(output, glasses_layer)


def _put_status(image: np.ndarray, text: str, color) -> None:
    h = image.shape[0]
    cv2.putText(image, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run(asset_dir: Path, camera_index: int = 0) -> None:
    """Run the interactive try-on loop until the user quits."""
    # Imported lazily so the rest of the package works without MediaPipe.
    import mediapipe as mp

    glasses_list: List[GlassesAsset] = load_all_glasses(asset_dir)
    glasses_idx = 0
    show_shadow = True

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {camera_index}). Check your camera.")

    print("Running... press 'q' to quit, 'n'/'p' to switch glasses, 's' to toggle shadow.")

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror view feels more natural
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    glasses = glasses_list[glasses_idx]
                    landmarks = result.multi_face_landmarks[0].landmark
                    display = render_glasses(frame, landmarks, glasses, show_shadow)
                    _put_status(display, f"Glasses: {glasses.name}", (0, 255, 0))
                else:
                    display = frame.copy()
                    _put_status(display, "No face detected", (0, 0, 255))

                cv2.imshow(WINDOW_TITLE, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), KEY_ESC):
                    break
                elif key == ord("n"):
                    glasses_idx = (glasses_idx + 1) % len(glasses_list)
                elif key == ord("p"):
                    glasses_idx = (glasses_idx - 1) % len(glasses_list)
                elif key == ord("s"):
                    show_shadow = not show_shadow
        finally:
            cap.release()
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glasses-tryon",
        description="Real-time virtual eyeglass try-on using your webcam.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory containing glasses PNG + JSON anchor files (default: %(default)s).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: %(default)s).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.assets_dir, args.camera)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
