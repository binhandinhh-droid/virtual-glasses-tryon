"""Remove the background of a glasses photo and save a transparent PNG.

Usage:
    glasses-remove-bg                          # file dialogs
    glasses-remove-bg INPUT_IMAGE [OUTPUT_PNG] # command line

Requires the optional ``rembg`` dependency:
    pip install "virtual-glasses-tryon[tools]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2

from .common import ask_directory, ask_open_file

IMAGE_FILETYPES = [("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]


def remove_background(image_bytes: bytes) -> bytes:
    """Return PNG bytes of ``image_bytes`` with the background removed."""
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            'rembg is not installed. Run: pip install "virtual-glasses-tryon[tools]"'
        ) from exc
    return remove(image_bytes)


def has_alpha_channel(path: Path) -> bool:
    """Check whether the image at ``path`` has a 4th (alpha) channel."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.ndim == 3 and image.shape[2] == 4


def _resolve_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path]]:
    input_path = args.input or ask_open_file("Step 1/2: Select the source image", IMAGE_FILETYPES)
    if not input_path:
        return None, None
    if args.output:
        return input_path, args.output

    output_dir = ask_directory("Step 2/2: Select the folder to save the result")
    if not output_dir:
        return input_path, None
    return input_path, output_dir / f"{input_path.stem}_nobg.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glasses-remove-bg",
        description="Remove the background of a glasses photo (outputs a transparent PNG).",
    )
    parser.add_argument("input", nargs="?", type=Path, help="Source image (JPG/PNG/...).")
    parser.add_argument("output", nargs="?", type=Path, help="Output PNG path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path, output_path = _resolve_paths(args)

    if not input_path or not input_path.exists():
        print("Error: source image not found or selection cancelled.", file=sys.stderr)
        return 1
    if not output_path:
        print("Error: no output location selected.", file=sys.stderr)
        return 1

    print(f"Reading image: {input_path}")
    image_bytes = input_path.read_bytes()

    print("Removing background... (the model is downloaded on first run)")
    try:
        result = remove_background(image_bytes)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result)

    if has_alpha_channel(output_path):
        print(f"Saved transparent image: {output_path}")
        print(f'Next step: glasses-punch-lens "{output_path}"  (or glasses-pick-anchors)')
        return 0

    print("Warning: the result may not contain an alpha channel; please check it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
