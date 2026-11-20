import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

from pipeline import paths
from pipeline.night_chalk_shader import build_hatching_bank, smoothstep

ROOT = paths.ROOT
REF = paths.CUBIST_REF

WARM = np.array([(94, 66, 34), (184, 92, 30), (205, 124, 62), (222, 154, 83), (233, 183, 116)], dtype=np.float32)
NEUT = np.array([(38, 74, 62), (74, 110, 96), (154, 168, 140), (226, 204, 155), (238, 228, 198)], dtype=np.float32)
COOL = np.array([(14, 22, 19), (24, 48, 42), (38, 74, 62), (74, 110, 96), (139, 168, 154)], dtype=np.float32)


def map_colors_smooth(colors):
    """Map source RGB colors smoothly onto reference hue ramps by continuous luminance."""
    c = colors / 255.0
    mx = c.max(axis=1)
    mn = c.min(axis=1)
    lum = 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]

    # Soft continuous weights across COOL / NEUT / WARM instead of hard discontinuous switches
    warmish = (r > b + 0.06) & (r >= g - 0.02)
    coolish = (b >= r - 0.01) | (g > r + 0.05)

    # Continuous position in [0, 4] along the 5-stop ramp (interpolated linearly)
    pos = np.clip(np.power(np.clip(lum, 0.0, 1.0), 0.8) * 4.0, 0.0, 4.0)
    idx0 = np.clip(np.floor(pos).astype(int), 0, 3)
    frac = (pos - idx0)[:, None]

    out = np.zeros((len(c), 3), dtype=np.float32)
    for i in range(len(c)):
        i0 = idx0[i]
        w_col = WARM[i0] * (1.0 - frac[i, 0]) + WARM[i0 + 1] * frac[i, 0]
        n_col = NEUT[i0] * (1.0 - frac[i, 0]) + NEUT[i0 + 1] * frac[i, 0]
        c_col = COOL[i0] * (1.0 - frac[i, 0]) + COOL[i0 + 1] * frac[i, 0]

        if mx[i] < 0.20:
            col = c_col
        elif sat[i] < 0.18:
            # Smooth transition between NEUT and warm/cool as sat approaches 0.18
            alpha_s = float(np.clip((sat[i] - 0.10) / 0.08, 0.0, 1.0))
            hue_col = w_col if warmish[i] else (c_col if coolish[i] else w_col)
            col = n_col * (1.0 - alpha_s) + hue_col * alpha_s
        elif warmish[i]:
            col = w_col
        elif coolish[i]:
            col = c_col
        else:
            col = w_col
        out[i] = col

    rl = (0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]) / 255.0
    scale = np.clip(lum / np.maximum(rl, 0.04), 0.82, 1.18)
    return out * scale[:, None]


def make_points(gray, spacing=170, seed=7):
    h, w = gray.shape
    rng = np.random.default_rng(seed)
    pts = []
    for y in range(0, h + spacing, spacing):
        for x in range(0, w + spacing, spacing):
            jx = rng.uniform(-0.30, 0.30) * spacing
            jy = rng.uniform(-0.30, 0.30) * spacing
            pts.append([min(max(x + jx, 0), w - 1), min(max(y + jy, 0), h - 1)])
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 93)
    cell = 88
    for cy in range(0, h, cell):
        for cx in range(0, w, cell):
            block = mag[cy : cy + cell, cx : cx + cell]
            if block.size and block.max() > thr:
                by, bx = np.unravel_index(np.argmax(block), block.shape)
                pts.append([float(cx + bx), float(cy + by)])
    cx0, cy0, rx, ry = 0.56 * w, 0.42 * h, 0.27 * w, 0.22 * h
    for y in range(int(cy0 - ry), int(cy0 + ry), 72):
        for x in range(int(cx0 - rx), int(cx0 + rx), 72):
            if ((x - cx0) / rx) ** 2 + ((y - cy0) / ry) ** 2 <= 1.0:
                pts.append([x + rng.uniform(-14, 14), y + rng.uniform(-14, 14)])
    for y in range(int(0.39 * h), int(0.48 * h), 34):
        for x in range(int(0.43 * w), int(0.72 * w), 34):
            pts.append([x + rng.uniform(-7, 7), y + rng.uniform(-7, 7)])
    pts += [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2]]
    return Delaunay(np.array(pts, dtype=np.float64))


def sample_triangle_colors(frame_rgb, tri):
    h, w, _ = frame_rgb.shape
    pts = tri.points
    simps = tri.simplices
    cents = pts[simps].mean(axis=1)
    cx = np.clip(cents[:, 0].astype(int), 0, w - 1)
    cy = np.clip(cents[:, 1].astype(int), 0, h - 1)
    cols = frame_rgb[cy, cx].astype(np.float32)

    verts = pts[simps].astype(int)
    vx = np.clip(verts[:, :, 0], 0, w - 1)
    vy = np.clip(verts[:, :, 1], 0, h - 1)
    vcols = frame_rgb[vy, vx].astype(np.float32).mean(axis=1)
    cols = 0.6 * cols + 0.4 * vcols

    vall = frame_rgb[vy, vx].astype(np.float32)
    dark = vall[np.arange(len(vall)), np.argmin(vall.sum(axis=2), axis=1)]
    cols = np.where((dark.sum(axis=1) / 3.0 < 70.0)[:, None], 0.45 * cols + 0.55 * dark, cols)
    return cols


def render_pose(frame_rgb, tri, cols_smoothed, hatch_map):
    h, w, _ = frame_rgb.shape
    img = Image.new("RGB", (w, h), (20, 24, 22))
    dr = ImageDraw.Draw(img)
    pts = tri.points
    simps = tri.simplices
    chosen = map_colors_smooth(cols_smoothed)
    fill = np.clip(0.92 * chosen + 0.08 * cols_smoothed, 0, 255).astype(int)
    for s, c in zip(simps, fill):
        poly = [tuple(pts[i]) for i in s]
        dr.polygon(poly, fill=tuple(c), outline=(28, 34, 26), width=2)
    out = np.asarray(img).astype(np.float32)

    # Subtle 12-fps chalk paper-tooth / directional pencil grain matching the main film
    lum_out = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    w_hatch = smoothstep(25.0, 60.0, lum_out) * (1.0 - smoothstep(185.0, 235.0, lum_out))
    out *= (1.0 + (0.065 * w_hatch * hatch_map)[:, :, None])

    # Continuous-tone XDoG + Sobel charcoal feature contours (glasses, eyes, beard, elevator)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), 0.75)
    g2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    dog = g1 - g2
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy) * 0.25

    dark_gate = 1.0 - smoothstep(145.0, 210.0, gray)
    ink = (
        np.clip((grad - 18.0) / 55.0, 0.0, 1.0) * 0.78
        + np.clip((-dog - 2.2) / 8.0, 0.0, 1.0) * 0.65
    ) * dark_gate
    ink = np.clip(ink, 0.0, 0.84)[:, :, None]
    out = out * (1.0 - ink) + np.array([16.0, 22.0, 19.0], dtype=np.float32) * ink
    return np.clip(out, 0, 255).astype(np.float32)


def read_frames_24fps(path, w=720, h=960):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps=24,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    frames = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy())
    proc.wait()
    return frames


def do_video(inp, outp, w=720, h=960):
    raw_frames = read_frames_24fps(inp, w, h)
    # Build one fixed Delaunay mesh over sampled 12-fps poses
    acc, cnt = None, 0
    for i in range(0, len(raw_frames), 24):
        g = raw_frames[i].mean(axis=2)
        acc = g if acc is None else acc + g
        cnt += 1
    tri = make_points(acc / max(cnt, 1))
    print(f"cubist_chalk triangles: {len(tri.simplices)} across {len(raw_frames)} frames ({cnt} sampled)")

    hatch_bank = build_hatching_bank(12, h, w, seed=99)
    rng = np.random.default_rng(23)

    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24",
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", "24", outp, "-y",
        ],
        stdin=subprocess.PIPE,
    )

    prev_cols = None
    cached_rgb = None
    for i, f_rgb in enumerate(raw_frames):
        pose_idx = i // 2
        if i % 2 == 0 or cached_rgb is None:
            cur_cols = sample_triangle_colors(f_rgb, tri)
            if prev_cols is None:
                cols_sm = cur_cols
            else:
                # Temporal EMA prevents isolated background wall triangle flashes
                cols_sm = 0.72 * cur_cols + 0.28 * prev_cols
            prev_cols = cols_sm
            cached_rgb = render_pose(f_rgb, tri, cols_sm, hatch_bank[pose_idx % len(hatch_bank)])

        out_f = cached_rgb + rng.normal(0.0, 2.5, (h, w, 1)).astype(np.float32)
        enc.stdin.write(np.clip(out_f, 0.0, 255.0).astype(np.uint8).tobytes())

    enc.stdin.close()
    enc.wait()
    print("wrote", outp, len(raw_frames), "frames")


if __name__ == "__main__":
    do_video(sys.argv[1], sys.argv[2])
