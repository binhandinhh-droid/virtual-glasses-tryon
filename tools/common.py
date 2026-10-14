"""Helpers shared by the asset preparation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


def ask_open_file(title: str, filetypes: Sequence[Tuple[str, str]]) -> Optional[Path]:
    """Show a native "open file" dialog. Returns ``None`` if cancelled or unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=title, filetypes=list(filetypes))
        root.destroy()
        return Path(selected) if selected else None
    except Exception:  # No display / tkinter missing: caller falls back to CLI args.
        return None


def ask_directory(title: str) -> Optional[Path]:
    """Show a native "choose folder" dialog. Returns ``None`` if cancelled or unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(selected) if selected else None
    except Exception:
        return None


def make_checkerboard(h: int, w: int, size: int = 16) -> np.ndarray:
    """Create a gray checkerboard (BGR) used to visualize transparency."""
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, size):
        for x in range(0, w, size):
            color = 200 if ((x // size) + (y // size)) % 2 == 0 else 150
            board[y : y + size, x : x + size] = color
    return board


def composite_over(rgba: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    """Composite a BGRA image over an opaque BGR background for display."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = rgba[:, :, :3].astype(np.float32)
    bg = background_bgr.astype(np.float32)
    return (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
