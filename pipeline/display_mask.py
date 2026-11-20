#!/usr/bin/env python3
"""Meme panels as content of the car's centre display, shown only during
real glances at the screen. Tracking features are seeded on the bezel/trim
ring around the screen (rigid with the display), never on the moving map
content, so the panel cannot drift relative to the screen.
Run via uv (opencv/numpy/pillow): track | seeds | apply <panel> <in> <out>"""
import json, os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths, quad_overlay as T

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
def R(x0, y0, x1, y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
# Narrow windows centred on verified seed frames: LK tracking only has to
# survive +-0.35s, during which the head is actually looking at the screen.
SPECS = [
 dict(panel=1, start=11.9, end=12.6, seed=12.25, kind="disp", inset=0.0,
      quad=[(475,626),(684,628),(683,704),(476,702)]),
 dict(panel=2, start=91.15, end=91.85, seed=91.5, kind="disp", inset=0.0,
      quad=[(506,636),(654,638),(653,714),(507,712)]),
 dict(panel=3, start=141.4, end=141.85, seed=141.5, kind="disp", inset=0.0,
      quad=[(425,540),(580,535),(578,615),(428,622)]),
 dict(panel=4, start=204.2, end=204.6, seed=204.25, kind="disp", inset=0.0,
      quad=[(465,645),(610,640),(608,725),(467,728)]),
 dict(panel=5, start=237.9, end=238.15, seed=238.0, kind="disp", inset=0.0,
      quad=[(563,582),(653,583),(652,658),(564,657)]),
 dict(panel=6, start=240.65, end=241.35, seed=241.0, kind="disp", inset=0.0,
      quad=[(590,596),(700,597),(699,674),(591,673)]),
]
T.SPECS = SPECS

TEXT = {2: ["THIS IS", "FINE."],
        3: ["I'M OKAY", "WITH THE", "EVENTS", "THAT ARE", "UNFOLDING", "CURRENTLY."],
        5: ["THAT'S OKAY,", "THINGS ARE", "GOING TO", "BE OKAY."]}

def _stylize_artwork(im):
    """Apply bilateral flattening, slight desaturation, step-28 posterization, and Canny ink lines."""
    im = ImageEnhance.Color(im).enhance(0.82)
    rgb = np.asarray(im, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bilat = cv2.bilateralFilter(bgr, 7, 36, 12)
    # RGB posterization matching build_v2.cartoon_body (step=28)
    post = ((bilat.astype(np.int32) // 28) * 28 + 14).clip(0, 255).astype(np.float32)
    gray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 70, 160), np.ones((2, 2), np.uint8)).astype(np.float32) / 255.0
    post *= (1.0 - 0.52 * edges[:, :, None])
    return Image.fromarray(cv2.cvtColor(np.clip(post, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

def display_face(spec):
    panel = spec["panel"]
    im = Image.open(os.path.join(T.MEME, f"panel{panel}.png")).convert("RGB")
    Wc, Hc = 720, 450
    if panel in TEXT:
        dog_w = 324
        crops = {
            2: (24, 96, im.width - 15, im.height),
            3: (20, 116, im.width - 15, im.height),
            5: (16, 76, im.width - 24, im.height),
        }
        dog = im.crop(crops[panel]).resize((dog_w, Hc), Image.LANCZOS)
        canvas = Image.new("RGB", (Wc, Hc), (232, 221, 174))
        canvas.paste(dog, (0, 0))
        im_art = _stylize_artwork(canvas)
        d = ImageDraw.Draw(im_art)
        # 4-px dark charcoal vertical divider between artwork and caption pane
        d.line([(dog_w, 0), (dog_w, Hc)], fill=(24, 22, 20), width=4)
        lines = TEXT[panel]
        size = {2: 92, 3: 48, 5: 48}[panel]
        line_gap = {2: 14, 3: 9, 5: 13}[panel]
        f = T.load_font(size)
        total_h = len(lines) * size + (len(lines) - 1) * line_gap
        y = (Hc - total_h) // 2
        # Center inside the inner pane (accounting for 14-px right border ring)
        cx = dog_w + (Wc - 14 - dog_w) // 2
        for ln in lines:
            d.text((cx, y), ln, font=f, fill=(20, 20, 18), anchor="ma")
            y += size + line_gap
        im = im_art
    else:
        crops = {
            1: (0, 34, im.width, 242),
            4: (0, 12, im.width, 252),
            6: (0, 20, im.width, 256),
        }
        band = im.crop(crops[panel]).resize((Wc, Hc), Image.LANCZOS)
        im = _stylize_artwork(band)
    im = im.filter(ImageFilter.GaussianBlur(0.85))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

T.panel_face = display_face

def seeds():
    for spec in SPECS:
        frames, fps = T.read_window(T.STORY, spec["start"], spec["end"])
        i = int(round((spec["seed"] - spec["start"]) * fps)); i = max(0, min(i, len(frames) - 1))
        diag_dir = os.path.join(paths.WORK, "track_diag"); os.makedirs(diag_dir, exist_ok=True)
        out = os.path.join(diag_dir, f"dseed_p{spec['panel']}.jpg")
        cv2.imwrite(out, frames[i]); print(spec["panel"], "idx", i, out)

if __name__ == "__main__":
    if sys.argv[1] == "manual":
        T.SPECS = SPECS; T.compute_tracks()
    elif sys.argv[1] == "track": T.compute_tracks()
    elif sys.argv[1] == "seeds": seeds()
    elif sys.argv[1] == "apply":
        with open(os.path.join(T.WORK, "track_quads.json")) as f: tracks = json.load(f)
        spec = next(s for s in SPECS if s["panel"] == int(sys.argv[2]))
        T.apply_one(spec, sys.argv[3], sys.argv[4], tracks)
