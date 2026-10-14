# Virtual Glasses Try-On

Real-time virtual eyeglass try-on from a webcam, built with
[MediaPipe Face Mesh](https://developers.google.com/mediapipe) and OpenCV.

## Features

- **Face landmark detection** with MediaPipe Face Mesh (468 landmarks).
- **Head roll compensation**: the glasses are fitted with a similarity transform
  (scale + rotation + translation) computed from the two eye positions.
- **Dynamic temple arms**: drawn as tapered curves from the frame hinges to the
  wearer's temples, so they follow face width and head tilt.
- **Soft drop shadow** under the frame for extra realism.
- **Asset pipeline** to turn any glasses photo into a try-on asset:
  background removal → semi-transparent lenses → anchor point picking.

## Installation

```bash
git clone https://github.com/<your-username>/virtual-glasses-tryon.git
cd virtual-glasses-tryon
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # core app
pip install -e ".[tools]"   # optional: background removal (rembg)
pip install -e ".[dev]"     # optional: pytest + ruff
```

Python 3.9+ and a webcam are required.

## Usage

### 1. Run the try-on

```bash
glasses-tryon --assets-dir assets/glasses --camera 0
# or: python -m glasses_tryon
```

| Key       | Action                         |
|-----------|--------------------------------|
| `n` / `p` | Next / previous glasses model  |
| `s`       | Toggle shadow                  |
| `q` / Esc | Quit                           |

### 2. Add your own glasses

Each model needs a transparent PNG and a JSON file with anchor points.
Run the tools in order (each opens a file dialog if you omit the path):

```bash
glasses-remove-bg photo.jpg photo_nobg.png     # 1. remove the background
glasses-punch-lens photo_nobg.png              # 2. make lenses semi-transparent
glasses-pick-anchors photo_nobg_lenspunch.png  # 3. click 4 anchor points
```

Copy the resulting `.png` and `.json` into `assets/glasses/`.

**Anchor click order:** left lens center → right lens center → left hinge → right hinge.

**Lens tool:** lens and frame are separated by *shape* (the frame is a thin rim,
the lens a thick region), not by color, so dark sunglass lenses also become
see-through. Tune "Frame thickness" and "Lens opacity" with the sliders.

## Project structure

```
virtual-glasses-tryon/
├── src/glasses_tryon/
│   ├── app.py            # webcam loop and CLI (glasses-tryon)
│   ├── asset.py          # GlassesAsset: PNG + anchor JSON loading
│   ├── geometry.py       # affine fitting, point transform, image warp
│   ├── temples.py        # dynamic temple arm rendering
│   ├── compositing.py    # alpha blending (Porter-Duff over)
│   ├── effects.py        # drop shadow
│   ├── lens.py           # shape-based lens transparency (pure functions)
│   ├── config.py         # constants and landmark indices
├── tools/            # asset preparation CLIs / GUIs
│   ├── remove_background.py
│   ├── punch_lens.py
│   ├── pick_anchors.py
│   └── common.py
├── assets/glasses/       # PNG + JSON pairs
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## Roadmap

- Occlusion handling (hands in front of the glasses)
- Full 3D head pose (yaw/pitch), not only roll
- Automatic brightness matching between glasses and scene

## License

MIT — see [LICENSE](LICENSE).
