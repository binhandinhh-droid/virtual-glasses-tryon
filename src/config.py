"""Shared constants and default settings."""

from pathlib import Path

# Default location of glasses assets (PNG + JSON pairs), relative to the CWD.
DEFAULT_ASSET_DIR = Path("assets") / "glasses"

# MediaPipe Face Mesh (468-point topology) landmark indices.
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263
# Temple-region landmarks (near the hairline / ear), on the same side as each eye.
LEFT_TEMPLE_IDX = 127
RIGHT_TEMPLE_IDX = 356

# Keys stored in the per-asset anchor JSON file.
ANCHOR_KEYS = (
    "left_eye_center",
    "right_eye_center",
    "left_temple_anchor",
    "right_temple_anchor",
)

# Rendering parameters.
TEMPLE_BASE_THICKNESS_PX = 6
TEMPLE_CURVE_SEGMENTS = 14
TEMPLE_CURVATURE = 0.12
TEMPLE_TAPER = 0.4
SHADOW_OFFSET = (3, 6)
SHADOW_STRENGTH = 0.35
SHADOW_BLUR_KSIZE = 9
