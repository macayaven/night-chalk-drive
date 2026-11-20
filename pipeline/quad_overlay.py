#!/usr/bin/env python3
"""Lock meme panels onto REAL signs in the footage.
For each appearance: seed the actual sign's corners in one frame, track the
sign plane with Lucas-Kanade optical flow + per-frame homography (forward and
backward from the seed), then warp the panel into the tracked quad on the
cartoon frames. The panel therefore inherits the signpost's exact motion and
perspective (zero drift relative to the landscape).
Run: uv run --with opencv-python-headless --with numpy --with pillow python pipeline.quad_overlay track|apply ...
"""
import json, os, subprocess, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths

ROOT = paths.ROOT
WORK = paths.WORK
MEME = os.path.join(WORK, "meme")
STORY = os.path.join(WORK, "story288.mp4")
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def load_font(size):
    for fp in (FONT, "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.isfile(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

SPECS = [
 dict(panel=1, start=16.7, end=17.5, seed=17.2, kind="wide", inset=0.06, expand=1.8,
      quad=[(205,407),(235,408),(235,432),(205,431)]),
 dict(panel=2, start=57.45, end=58.4, seed=58.0, kind="text2", inset=0.01, expand=1.5,
      quad=[(527,463),(573,465),(572,502),(528,500)]),
 dict(panel=3, start=82.75, end=83.2, seed=82.8, kind="vtext", inset=0.04,
      quad=[(545,475),(578,476),(577,522),(545,520)]),
 dict(panel=4, start=203.25, end=204.0, seed=203.25, kind="full", inset=0.02, expand=1.3,
      quad=[(562,472),(630,474),(630,512),(562,510)]),
 dict(panel=5, start=258.6, end=260.8, seed=260.6, kind="full", inset=0.0,
      quad=[(115,635),(335,620),(345,845),(125,870)],
      keys=[(258.6,[(285,320),(500,295),(475,545),(300,580)]),
            (259.0,[(130,350),(315,325),(345,565),(155,605)]),
            (259.4,[(245,350),(450,325),(455,565),(260,600)]),
            (259.8,[(245,350),(450,325),(450,565),(260,600)]),
            (260.2,[(95,515),(320,490),(330,720),(105,760)]),
            (260.6,[(115,635),(335,620),(345,845),(125,870)]),
            (260.8,[(105,640),(320,625),(330,850),(115,875)])]),
 dict(panel=6, start=268.0, end=269.3, seed=268.5, kind="full", inset=0.03,
      quad=[(304,219),(395,224),(390,348),(309,341)]),
]

def panel_face(spec):
    im = Image.open(os.path.join(MEME, f"panel{spec['panel']}.png")).convert("RGB")
    kind = spec["kind"]
    if kind == "wide":  # image-only panel into a wide sign: central dog/flames band
        im = im.crop((0, 45, im.width, 230)).resize((640, 420), Image.LANCZOS)
    elif kind == "text2":  # short caption, redrawn big so it reads at sign size
        im = im.resize((640, 620), Image.LANCZOS)
        d = ImageDraw.Draw(im); f = load_font(104)
        d.rectangle([0, 0, 640, 250], fill=(246, 239, 190))
        d.text((320, 58), "THIS IS", font=f, fill=(10, 10, 10), anchor="ma")
        d.text((320, 158), "FINE.", font=f, fill=(10, 10, 10), anchor="ma")
    elif kind == "vtext":  # long caption into a vertical bus-shelter ad face
        canvas = Image.new("RGB", (480, 660), (246, 239, 190))
        d = ImageDraw.Draw(canvas); f = load_font(57)
        lines = ["I'M OKAY", "WITH THE", "EVENTS", "THAT ARE", "UNFOLDING", "CURRENTLY."]
        for i, ln in enumerate(lines):
            d.text((240, 18 + i * 62), ln, font=f, fill=(10, 10, 10), anchor="ma")
        dog = im.crop((40, 118, im.width - 20, im.height)).resize((480, 252), Image.LANCZOS)
        canvas.paste(dog, (0, 408)); im = canvas
    else:
        im = im.resize((560, 542), Image.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(1.07).filter(ImageFilter.UnsharpMask(1.2, 150, 2))
    arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return cv2.copyMakeBorder(arr, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(12, 12, 12))

def read_window(path, start, end):
    # Decode via ffmpeg accurate seeking so frame indices align exactly with
    # the ffmpeg-extracted segments the quads are later applied to.
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    cmd = ["ffmpeg","-v","error"]
    if start: cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if end < 1e8: cmd += ["-t", str(end - start)]
    cmd += ["-vf","fps=24","-f","rawvideo","-pix_fmt","bgr24","-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3; frames = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n: break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy())
    proc.wait(); return frames, 24.0

def expand_quad(q, k):
    c = q.mean(axis=0); return c + (q - c) * k

def track_one(spec, frames, fps):
    if "keys" in spec:  # hand-keyframed plane (feature-poor door surface)
        keys = [(t, np.array(q, dtype=np.float32)) for t, q in spec["keys"]]
        out = []
        for i in range(len(frames)):
            t = spec["start"] + i / fps
            if t <= keys[0][0]: q = keys[0][1]
            elif t >= keys[-1][0]: q = keys[-1][1]
            else:
                for (t0, q0), (t1, q1) in zip(keys, keys[1:]):
                    if t0 <= t <= t1: q = q0 + (q1 - q0) * ((t - t0) / (t1 - t0)); break
            out.append(q.tolist())
        print(f"panel{spec['panel']}: manual keyframes, {len(frames)} frames")
        return out
    seed_i = int(round((spec["seed"] - spec["start"]) * fps))
    seed_i = max(0, min(seed_i, len(frames) - 1))
    quad0 = np.array(spec["quad"], dtype=np.float32)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    mask = np.zeros(grays[0].shape, dtype=np.uint8)
    cv2.fillPoly(mask, [expand_quad(quad0, spec.get("expand", 2.1)).astype(np.int32)], 255)
    if spec.get("ring"):  # track the bezel/trim around a screen, not its moving content
        inner = np.zeros_like(mask)
        cv2.fillPoly(inner, [expand_quad(quad0, 0.90).astype(np.int32)], 255)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner))
    pts = cv2.goodFeaturesToTrack(grays[seed_i], 260, 0.006, 3, mask=mask, blockSize=3)
    if pts is None or len(pts) < 10:
        pts = cv2.goodFeaturesToTrack(grays[seed_i], 260, 0.002, 2, mask=mask, blockSize=3)
    seed_pts = pts.reshape(-1, 2).astype(np.float32)
    print(f"panel{spec['panel']}: {len(seed_pts)} seed features, {len(frames)} frames")
    quads = [None] * len(frames); quads[seed_i] = quad0.tolist()
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for direction, rng in ((1, range(seed_i + 1, len(frames))), (-1, range(seed_i - 1, -1, -1))):
        sp = seed_pts.copy(); prev = seed_pts.copy(); lastH = np.eye(3, dtype=np.float64)
        prev_i = seed_i
        for i in rng:
            cur, st, _ = cv2.calcOpticalFlowPyrLK(grays[prev_i], grays[i], prev.reshape(-1, 1, 2), None, **lk)
            cur = cur.reshape(-1, 2); ok = st.reshape(-1) == 1
            if ok.sum() >= 8:
                sp, prev = sp[ok], cur[ok]
                H, inl = cv2.findHomography(sp, prev, cv2.RANSAC, 2.5)
                if H is not None and inl is not None and int(inl.sum()) >= 6: lastH = H
            q = cv2.perspectiveTransform(quad0.reshape(1, -1, 2), lastH).reshape(-1, 2)
            quads[i] = q.tolist(); prev_i = i
    return quads

def compute_tracks():
    out = {}
    diag_dir = os.path.join(WORK, "track_diag"); os.makedirs(diag_dir, exist_ok=True)
    for spec in SPECS:
        frames, fps = read_window(STORY, spec["start"], spec["end"])
        quads = track_one(spec, frames, fps)
        out[str(spec["panel"])] = dict(start=spec["start"], fps=fps, quads=quads)
        imgs = []
        for idx in (0, len(frames) // 2, len(frames) - 1):
            f = frames[idx].copy()
            cv2.polylines(f, [np.array(quads[idx]).astype(np.int32)], True, (0, 0, 255), 3)
            imgs.append(cv2.resize(f, (360, 480)))
        cv2.imwrite(os.path.join(diag_dir, f"panel{spec['panel']}.jpg"), np.hstack(imgs))
    path = os.path.join(WORK, "track_quads.json")
    with open(path, "w") as f: json.dump(out, f)
    print("wrote", path)

def _rounded_mask(w, h, r, pad=0):
    """Build a 2x-supersampled anti-aliased rounded-rectangle mask."""
    sw, sh = w * 2, h * 2
    p, rr = pad * 2, max(2, r * 2)
    mask = np.zeros((sh, sw), dtype=np.uint8)
    x0, y0, x1, y1 = p, p, sw - 1 - p, sh - 1 - p
    if x1 <= x0 + 2 * rr or y1 <= y0 + 2 * rr:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    else:
        cv2.rectangle(mask, (x0 + rr, y0), (x1 - rr, y1), 255, -1)
        cv2.rectangle(mask, (x0, y0 + rr), (x1, y1 - rr), 255, -1)
        for cx, cy in [(x0 + rr, y0 + rr), (x1 - rr, y0 + rr), (x1 - rr, y1 - rr), (x0 + rr, y1 - rr)]:
            cv2.circle(mask, (cx, cy), rr, 255, -1)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

def apply_one(spec, inp, outp, tracks):
    frames, fps = read_window(inp, 0, 10 ** 9)
    tr = tracks[str(spec["panel"])]; quads = tr["quads"]
    face = panel_face(spec); fh, fw = face.shape[:2]
    src = np.array([(0, 0), (fw - 1, 0), (fw - 1, fh - 1), (0, fh - 1)], dtype=np.float32)
    # Rounded-corner outer mask and 14-px inset inner mask so the tile carries a
    # perspective-foreshortened cartoon dark ink bezel border.
    r = max(6, int(round(fh * 58.0 / 450.0)))
    outer_mask = _rounded_mask(fw, fh, r, pad=0)
    inner_mask = _rounded_mask(fw, fh, max(4, r - 14), pad=14)
    border_alpha = np.clip((outer_mask.astype(np.float32) - inner_mask.astype(np.float32)) / 255.0, 0.0, 1.0)[:, :, None]
    ink_bgr = np.array([16.0, 18.0, 16.0], dtype=np.float32)
    face = np.clip(face.astype(np.float32) * (1.0 - border_alpha) + ink_bgr * border_alpha, 0, 255).astype(np.uint8)
    enc = subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","bgr24","-s","720x960",
                            "-r","24","-i","-","-an","-c:v","libx264","-crf","14","-preset","veryfast",
                            "-pix_fmt","yuv420p","-r","24",outp,"-y"], stdin=subprocess.PIPE)
    rng = np.random.default_rng(11)
    for i, frame in enumerate(frames):
        qi = quads[min(i, len(quads) - 1)]
        if qi is not None:
            q = np.array(qi, dtype=np.float32)
            if spec["inset"]: q = expand_quad(q, 1 - spec["inset"] * 2)
            M = cv2.getPerspectiveTransform(src, q)
            warped = cv2.warpPerspective(face, M, (720, 960), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            warped = cv2.GaussianBlur(warped, (0, 0), 0.65)
            m_raw = cv2.warpPerspective(outer_mask, M, (720, 960), flags=cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m_raw, (0, 0), 0.65).astype(np.float32) / 255.0
            m_inner = cv2.warpPerspective(inner_mask, M, (720, 960), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            sel = m_inner > 0.5
            if not sel.any():
                sel = m > 0.5
            if sel.any():
                env = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (31, 31), 10)
                lum_sign = float(env[sel].mean())
                lum_face = float(warped[sel].mean()) + 1e-6
                base_gain = min(max((lum_sign + 18.0) / lum_face, 0.58), 0.88)
                spatial_mod = np.clip(0.84 + 0.16 * (env / (lum_sign + 1e-6)), 0.72, 1.15)[:, :, None]
                n_luma = rng.normal(0.0, 6.0, (960, 720, 1)).astype(np.float32)
                wf = warped.astype(np.float32) * base_gain * spatial_mod + n_luma
                a = (m * 0.98)[:, :, None]
                frame = np.clip(frame.astype(np.float32) * (1.0 - a) + np.clip(wf, 0.0, 255.0) * a, 0.0, 255.0).astype(np.uint8)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close(); enc.wait()
    print("treated", outp, len(frames), "frames")

if __name__ == "__main__":
    if sys.argv[1] == "track":
        compute_tracks()
    elif sys.argv[1] == "apply":  # apply <panel> <input> <output>
        with open(os.path.join(WORK, "track_quads.json")) as f: tracks = json.load(f)
        spec = next(s for s in SPECS if s["panel"] == int(sys.argv[2]))
        apply_one(spec, sys.argv[3], sys.argv[4], tracks)
