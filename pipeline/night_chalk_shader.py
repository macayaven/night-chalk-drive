#!/usr/bin/env python3
"""Take On Me Night-Chalk Rotoscope engine + single-pass streaming renderer.

Transforms real night-drive footage (work/final3_parts/head_raw.mp4) at 12-fps
animation cadence into a sharp, high-temperature, high-contrast chalk-palette
rotoscope film that keeps dark night skies clean and real-world road/cabin
details crisp, while compositing the six perspective-locked centre-display
replacements in the same pass.

Usage:
  uv run --with opencv-python-headless --with numpy --with pillow \
      python -m pipeline.night_chalk_shader preview
  uv run --with opencv-python-headless --with numpy --with pillow \
      python -m pipeline.night_chalk_shader render <in_head_raw> <out_head_signed>
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import display_mask as D, media_base as B, paths, quad_overlay as T

W, H = 720, 960
HEAD_RAW = os.path.join(B.WORK, "final3_parts", "head_raw.mp4")
HEAD_CARTOON_PREV = os.path.join(B.WORK, "final3_parts", "head_cartoon.mp4")

HEAD_SIGNS = [
    (1, 11.9, 12.6),
    (2, 91.15, 91.85),
    (3, 141.4, 141.85),
    (4, 204.2, 204.6),
    (5, 237.9, 238.15),
    (6, 240.65, 241.35),
]


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_line_kernel(length, angle_deg):
    k = np.zeros((length, length), dtype=np.float32)
    c = length // 2
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    for t in np.linspace(-c, c, length * 3):
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def build_hatching_bank(n_variants=12, h=H, w=W, seed=42):
    """Precompute 12 crisp directional (+45 deg / -35 deg) rotoscope pencil hatching maps."""
    rng = np.random.default_rng(seed)
    k45 = make_line_kernel(17, -45.0)
    km35 = make_line_kernel(11, 35.0)
    bank = []
    for _ in range(n_variants):
        n1 = rng.normal(0.0, 1.0, (h // 2, w // 2)).astype(np.float32)
        n1 = cv2.resize(n1, (w, h), interpolation=cv2.INTER_NEAREST)
        n2 = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        h1 = cv2.filter2D(n1, -1, k45)
        h2 = cv2.filter2D(n1, -1, km35)
        tooth = cv2.GaussianBlur(n2, (0, 0), 0.7)
        combined = 0.65 * h1 + 0.25 * h2 + 0.14 * tooth
        combined = (combined - combined.mean()) / (combined.std() + 1e-6)
        bank.append(np.clip(combined, -2.2, 2.2).astype(np.float32))
    return bank


class ChalkRotoscopeEngine:
    """CIE-Lab night-chalk rotoscope shader operating at 12-fps cadence."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        self.hatch_bank = build_hatching_bank(12, H, W)
        self.xp = np.array([0.0, 14.0, 22.0, 42.0,  85.0, 145.0, 210.0, 255.0], dtype=np.float32)
        self.fp = np.array([0.0,  4.0,  9.0, 44.0, 106.0, 184.0, 244.0, 255.0], dtype=np.float32)

    def render_pose(self, bgr, pose_idx, palette=None):
        """Render a single 12-fps rotoscope pose.

        Optional palette: (a_warm, b_warm, a_cool, b_cool, rim_bgr_tuple).
        """
        if palette is None:
            a_warm, b_warm, a_cool, b_cool = 7.5, 16.5, -3.0, -5.0
            rim_bgr = np.array([165.0, 224.0, 250.0], dtype=np.float32)
        else:
            a_warm, b_warm, a_cool, b_cool, rim_t = palette
            rim_bgr = np.array(rim_t, dtype=np.float32)
        # 1. Three-pass edge-preserving bilateral smoothing for solid painterly chalk planes
        sm = cv2.bilateralFilter(bgr, 9, 45, 10)
        sm = cv2.bilateralFilter(sm, 7, 38, 8)
        sm = cv2.bilateralFilter(sm, 7, 32, 8)

        # 2. CIE Lab conversion & night-black / sculpted chalk tonal modeling
        lab = cv2.cvtColor(sm, cv2.COLOR_BGR2Lab)
        L_u8, a_u8, b_u8 = cv2.split(lab)
        L = L_u8.astype(np.float32)
        L_clahe = self.clahe.apply(L_u8).astype(np.float32)
        L_base = 0.70 * L + 0.30 * L_clahe

        night_gate = smoothstep(14.0, 28.0, L)
        L_curve = np.interp(L_base, self.xp, self.fp).astype(np.float32)

        # Soft-tanh chalk tonal bands (sculpted illustration planes without hard blobs)
        step = 26.0
        q_idx = np.floor(L_curve / step)
        phase = (L_curve / step) - q_idx - 0.5
        L_quant = (q_idx + 0.5 + 0.5 * np.tanh(phase * 4.2) / np.tanh(2.1)) * step
        L_toned = L_curve * (1.0 - 0.58 * night_gate) + L_quant * (0.58 * night_gate)
        L_toned = np.clip(L_toned, 0.0, 255.0)

        # 3. High-temperature warm chalk palette + subtle cool slate shadows, neutral black sky
        a_off = a_u8.astype(np.float32) - 128.0
        b_off = b_u8.astype(np.float32) - 128.0
        C = np.hypot(a_off, b_off)
        sat_boost = 1.92 * smoothstep(15.0, 34.0, L)
        neutral_mask = 1.0 - 0.50 * np.clip((C - 28.0) / 38.0, 0.0, 1.0)
        warm_w = smoothstep(26.0, 72.0, L_toned) * neutral_mask
        cool_w = smoothstep(15.0, 28.0, L) * (1.0 - smoothstep(26.0, 56.0, L_toned)) * neutral_mask

        a_new = np.clip(128.0 + a_off * sat_boost + a_warm * warm_w + a_cool * cool_w, 0.0, 255.0)
        b_new = np.clip(128.0 + b_off * sat_boost + b_warm * warm_w + b_cool * cool_w, 0.0, 255.0)

        graded_lab = cv2.merge([
            L_toned.astype(np.uint8),
            a_new.astype(np.uint8),
            b_new.astype(np.uint8),
        ])
        chalk_bgr = cv2.cvtColor(graded_lab, cv2.COLOR_Lab2BGR).astype(np.float32)

        # 4. 12-fps directional pencil hatching in shadow-to-midtone gradients only
        hatch = self.hatch_bank[pose_idx % len(self.hatch_bank)]
        w_hatch = smoothstep(20.0, 40.0, L_toned) * (1.0 - smoothstep(115.0, 175.0, L_toned))
        chalk_bgr *= (1.0 + (0.075 * w_hatch * hatch)[:, :, None])

        # 5. Confident continuous-tone charcoal rotoscope linework + dry chalk rim highlights
        g1 = cv2.GaussianBlur(L_base, (0, 0), 0.75)
        g2 = cv2.GaussianBlur(L_base, (0, 0), 2.10)
        dog = g1 - g2
        sx = cv2.Sobel(L_base, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(L_base, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sx, sy) * 0.25

        sky_gate = smoothstep(15.0, 27.0, L)
        ink_raw = (
            np.clip((grad - 13.0) / 42.0, 0.0, 1.0) * 0.85
            + np.clip((-dog - 1.6) / 6.5, 0.0, 1.0) * 0.72
        )
        ink_bold = 0.65 * ink_raw + 0.35 * cv2.dilate(ink_raw, np.ones((2, 2), np.uint8))
        ink = np.clip(ink_bold * sky_gate, 0.0, 0.90)
        chalk_bgr *= (1.0 - ink[:, :, None])

        # Dry warm cream-gold chalk rim highlight along lit contours
        rim = np.clip((dog - 2.6) / 10.0, 0.0, 1.0) * smoothstep(50.0, 105.0, L_toned)
        chalk_bgr += rim[:, :, None] * rim_bgr * 0.32

        return np.clip(chalk_bgr, 0.0, 255.0).astype(np.float32)


def prepare_display_assets():
    """Prepare chalk-matched display faces, rounded masks, and tracked quad lookup."""
    q_work = os.path.join(B.WORK, "track_quads.json")
    q_path = q_work if os.path.isfile(q_work) else paths.QUADS_EXPEDITION
    with open(q_path) as f:
        tracks = json.load(f)

    faces = {}
    for spec in D.SPECS:
        p = spec["panel"]
        face = D.display_face(spec)
        fh, fw = face.shape[:2]
        r = max(6, int(round(fh * 58.0 / 450.0)))
        outer_mask = T._rounded_mask(fw, fh, r, pad=0)
        inner_mask = T._rounded_mask(fw, fh, max(4, r - 14), pad=14)
        border_alpha = np.clip((outer_mask.astype(np.float32) - inner_mask.astype(np.float32)) / 255.0, 0.0, 1.0)[:, :, None]
        ink_bgr = np.array([16.0, 18.0, 16.0], dtype=np.float32)
        face = np.clip(face.astype(np.float32) * (1.0 - border_alpha) + ink_bgr * border_alpha, 0, 255).astype(np.uint8)
        faces[p] = (face, outer_mask, inner_mask)

    # Map global frame indices (exact match to v1 glance splice offsets) -> (panel, local_idx, total_len)
    frame_map = {}
    for p, st, en in HEAD_SIGNS:
        start_f = int(round(st * 24.0))
        quads = tracks[str(p)]["quads"]
        total_len = len(quads)
        for local_i, q in enumerate(quads):
            if q is not None:
                frame_map[start_f + local_i] = (p, local_i, total_len, np.array(q, dtype=np.float32))
    return faces, frame_map


def composite_tile(frame_f32, panel_data, face_tuple, hatch_map, rng):
    """Warp and composite a chalk-matched screen replacement onto a float32 frame."""
    p, local_i, total_len, q = panel_data
    face, outer_mask, inner_mask = face_tuple
    fh, fw = face.shape[:2]
    src = np.array([(0, 0), (fw - 1, 0), (fw - 1, fh - 1), (0, fh - 1)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, q)

    warped = cv2.warpPerspective(face, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped = cv2.GaussianBlur(warped, (0, 0), 0.65).astype(np.float32)
    m_raw = cv2.warpPerspective(outer_mask, M, (W, H), flags=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m_raw, (0, 0), 0.65).astype(np.float32) / 255.0
    m_inner = cv2.warpPerspective(inner_mask, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    sel = m_inner > 0.5
    if not sel.any():
        sel = m > 0.5
    if not sel.any():
        return frame_f32

    env = cv2.GaussianBlur(cv2.cvtColor(np.clip(frame_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (31, 31), 10)
    lum_sign = float(env[sel].mean())
    lum_face = float(warped[sel].mean()) + 1e-6
    base_gain = min(max((lum_sign + 20.0) / lum_face, 0.58), 0.88)
    spatial_mod = np.clip(0.84 + 0.16 * (env / (lum_sign + 1e-6)), 0.72, 1.15)[:, :, None]

    wf = warped * base_gain * spatial_mod
    # Subtle 12-fps chalk paper tooth over the screen so it sits natively in the rotoscope frame
    wf *= (1.0 + 0.045 * hatch_map[:, :, None])
    wf += rng.normal(0.0, 3.0, (H, W, 1)).astype(np.float32)

    # Full opaque replacement on every frame (no ghosting of underlying CarPlay/camera UI)
    a = (m * 0.98)[:, :, None]
    return frame_f32 * (1.0 - a) + np.clip(wf, 0.0, 255.0) * a


def grab_frame_at(path, t):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read()
    return np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()


def run_preview():
    engine = ChalkRotoscopeEngine()
    faces, frame_map = prepare_display_assets()
    rng = np.random.default_rng(11)

    # Key scenes: garage/P1 (12.25s), street/crosswalk (55.0s), lights/P2 (91.5s), city/P3 (141.5s), gate/P4 (204.35s)
    scenes = [
        ("garage + P1", 12.25, 1),
        ("street 55s", 55.00, None),
        ("lights + P2", 91.50, 2),
        ("city + P3", 141.50, 3),
        ("gate + P4", 204.35, 4),
    ]

    rows_raw, rows_prev, rows_ours = [], [], []
    # Also check head_sign{p}.mp4 mid frame for prev model comparison
    for idx, (label, t, p) in enumerate(scenes):
        raw_fr = grab_frame_at(HEAD_RAW, t)
        if p is not None:
            # Find frame matching t in head_sign{p} (from v1 fixed tiles or cartoon)
            g_idx = int(round(t * 24.0))
            pose_idx = g_idx // 2
            prev_fr = grab_frame_at(HEAD_CARTOON_PREV, t)
        else:
            g_idx = int(round(t * 24.0))
            pose_idx = g_idx // 2
            prev_fr = grab_frame_at(HEAD_CARTOON_PREV, t)

        ours_f = engine.render_pose(raw_fr, pose_idx)
        if g_idx in frame_map:
            pd = frame_map[g_idx]
            ours_f = composite_tile(ours_f, pd, faces[pd[0]], engine.hatch_bank[pose_idx % 12], rng)
        ours_u8 = np.clip(ours_f + rng.normal(0.0, 2.5, (H, W, 1)).astype(np.float32), 0.0, 255.0).astype(np.uint8)

        def label_tile(im, txt):
            r = cv2.resize(im, (360, 480))
            cv2.putText(r, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 180), 2)
            return r

        rows_raw.append(label_tile(raw_fr, f"RAW: {label}"))
        rows_prev.append(label_tile(prev_fr, f"OLD CARTOON: {label}"))
        rows_ours.append(label_tile(ours_u8, f"CHALK ROTO: {label}"))

    grid = np.vstack([np.hstack(rows_raw), np.hstack(rows_prev), np.hstack(rows_ours)])
    out_path = os.path.join(B.WORK, "track_diag", "new_style_comparison.jpg")
    cv2.imwrite(out_path, grid)
    print("Wrote", out_path)


def run_render(inp, outp):
    engine = ChalkRotoscopeEngine()
    faces, frame_map = prepare_display_assets()
    rng = np.random.default_rng(11)

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", inp, "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "24",
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", "24", outp, "-y",
        ],
        stdin=subprocess.PIPE,
    )

    n = W * H * 3
    i = 0
    cached_pose = None
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        raw_fr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
        pose_idx = i // 2
        if i % 2 == 0 or cached_pose is None:
            cached_pose = engine.render_pose(raw_fr, pose_idx)
        out_f = cached_pose.copy()
        if i in frame_map:
            pd = frame_map[i]
            out_f = composite_tile(out_f, pd, faces[pd[0]], engine.hatch_bank[pose_idx % 12], rng)
        out_f += rng.normal(0.0, 2.5, (H, W, 1)).astype(np.float32)
        enc.stdin.write(np.clip(out_f, 0.0, 255.0).astype(np.uint8).tobytes())
        i += 1
        if i % 480 == 0:
            print(f"rendered {i} frames ({i/24.0:.1f}s)", flush=True)

    dec.wait()
    enc.stdin.close()
    enc.wait()
    print(f"Finished rendering {i} frames -> {outp}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if mode == "preview":
        run_preview()
    elif mode == "render":
        run_render(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"Unknown mode: {mode}")
