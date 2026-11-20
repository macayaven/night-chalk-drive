#!/usr/bin/env python3
"""Render narrative-aligned 12-fps Cubist-Chalk elevator protagonist endings
(tail_{nansen,shackleton,mars500}.mp4) + enhanced Intro/Outro story cards.

Each 516-frame (21.50s) tail:
1. Opens seamlessly through the Cubist-styled elevator doorway & button press
   from work/final3_parts/tail_raw.mp4 (f0..f78) using that story's custom
   5-stop WARM/NEUT/COOL Cubist color ramp.
2. Cross-dissolves (f54..f78) into the commissioned inside-elevator Cubist
   protagonist painting aligned with the narrative:
     - Nansen     : Arctic Fram frosted-ice cabin + aurora-cyan shimmer
     - Shackleton : 1915 Endurance timber/bronze cabin + warm amber lantern flicker
     - Mars500    : Sealed IBMP spacecraft airlock warming from cyan -> gold
3. Burns the lower-third Outro card (t_tail 4.2..19.2s) with headline,
   attribution, and the short paragraph explaining why that story is worth telling.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import cubist_portrait as CC, fetch_archival_media as FEA, media_base as B, night_chalk_shader as R

W, H = R.W, R.H
TAIL_RAW = os.path.join(B.WORK, "final3_parts", "tail_raw.mp4")
PROTAG_DIR = os.path.join(B.WORK, "trilogy_assets", "protagonists")
TRI_DIR = os.path.join(B.WORK, "trilogy_assets", "films")

# Custom Cubist doorway/entry color ramps per narrative (RGB)
CUBIST_RAMPS = {
    "nansen": {
        "WARM": np.array([(96, 132, 152), (145, 185, 206), (195, 215, 212), (226, 222, 202), (242, 246, 248)], dtype=np.float32),
        "NEUT": np.array([(26, 52, 68), (58, 96, 118), (125, 168, 188), (196, 224, 236), (235, 246, 252)], dtype=np.float32),
        "COOL": np.array([(12, 24, 38), (24, 52, 76), (48, 94, 122), (88, 152, 178), (146, 212, 228)], dtype=np.float32),
        "ink": (14, 26, 36),
    },
    "shackleton": {
        "WARM": CC.WARM,
        "NEUT": CC.NEUT,
        "COOL": CC.COOL,
        "ink": (28, 34, 26),
    },
    "mars500": {
        "WARM": np.array([(78, 62, 54), (156, 98, 64), (198, 134, 78), (225, 174, 108), (242, 218, 165)], dtype=np.float32),
        "NEUT": np.array([(22, 34, 62), (48, 72, 112), (112, 148, 175), (186, 214, 224), (228, 240, 244)], dtype=np.float32),
        "COOL": np.array([(10, 18, 42), (18, 38, 82), (32, 82, 118), (58, 148, 156), (112, 216, 208)], dtype=np.float32),
        "ink": (14, 20, 32),
    },
}


def fit_font(dr, text, max_w, max_size):
    for sz in range(max_size, 12, -1):
        f = FEA.load_font(sz)
        tb = dr.textbbox((0, 0), text, font=f)
        if (tb[2] - tb[0]) <= max_w:
            return f, tb[2] - tb[0]
    f = FEA.load_font(12)
    tb = dr.textbbox((0, 0), text, font=f)
    return f, tb[2] - tb[0]


def wrap_paragraph(dr, text, max_w, max_lines=5, start_size=16, min_size=13):
    """Word-wrap paragraph to fit max_w and <= max_lines, returning (font, lines, line_h)."""
    for sz in range(start_size, min_size - 1, -1):
        font = FEA.load_font(sz)
        words = text.split()
        lines = []
        cur = ""
        for word in words:
            cand = word if not cur else f"{cur} {word}"
            tb = dr.textbbox((0, 0), cand, font=font)
            if (tb[2] - tb[0]) <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines:
            return font, lines, sz + 6
    font = FEA.load_font(min_size)
    words = text.split()
    lines, cur = [], ""
    for word in words:
        cand = word if not cur else f"{cur} {word}"
        if dr.textbbox((0, 0), cand, font=font)[2] <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return font, lines, min_size + 5


def render_story_card(card_tuple, accent_rgb, is_intro=True):
    """Render an auto-wrapped Intro or Outro story card (BGR float32, Alpha float32 HxWx1)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    margin_x = 26
    pad_x = 22
    max_w = (W - 2 * margin_x) - 2 * pad_x

    if is_intro:
        kicker, title, subtitle, paragraph = card_tuple
        f_kick, _ = fit_font(dr, kicker, max_w, 17)
        f_title, _ = fit_font(dr, title, max_w, 32)
        f_sub, _ = fit_font(dr, subtitle, max_w, 18)
        f_para, lines, line_h = wrap_paragraph(dr, paragraph, max_w, max_lines=5, start_size=16, min_size=13)

        y0 = 44
        div_y = y0 + 110
        para_y0 = div_y + 12
        y1 = para_y0 + len(lines) * line_h + 14

        dr.rectangle([margin_x, y0, W - margin_x, y1], fill=(8, 12, 18, 212))
        dr.rectangle([margin_x, y0, margin_x + 6, y1], fill=(*accent_rgb, 255))
        dr.line([(margin_x, y0), (W - margin_x, y0)], fill=(*accent_rgb, 195), width=2)
        dr.line([(margin_x, y1), (W - margin_x, y1)], fill=(*accent_rgb, 195), width=2)
        dr.line([(margin_x + pad_x, div_y), (W - margin_x - pad_x, div_y)], fill=(*accent_rgb, 120), width=1)

        tx = margin_x + pad_x
        dr.text((tx, y0 + 16), kicker, font=f_kick, fill=(*accent_rgb, 250))
        dr.text((tx, y0 + 42), title, font=f_title, fill=(248, 250, 255, 255))
        dr.text((tx, y0 + 82), subtitle, font=f_sub, fill=(208, 222, 238, 240))
        for li, ln in enumerate(lines):
            dr.text((tx, para_y0 + li * line_h), ln, font=f_para, fill=(234, 241, 249, 248))
    else:
        headline, attrib, paragraph = card_tuple
        f_head, w_h = fit_font(dr, headline, max_w, 31)
        f_attr, w_a = fit_font(dr, attrib, max_w, 18)
        f_para, lines, line_h = wrap_paragraph(dr, paragraph, max_w, max_lines=5, start_size=16, min_size=13)

        y1 = 928
        para_h = len(lines) * line_h
        y0 = y1 - (92 + para_h + 14)
        div_y = y0 + 80
        para_y0 = div_y + 12

        dr.rectangle([margin_x, y0, W - margin_x, y1], fill=(8, 11, 16, 224))
        dr.rectangle([margin_x, y0, margin_x + 6, y1], fill=(*accent_rgb, 255))
        dr.line([(margin_x, y0), (W - margin_x, y0)], fill=(*accent_rgb, 220), width=2)
        dr.line([(margin_x, y1), (W - margin_x, y1)], fill=(*accent_rgb, 220), width=2)
        dr.line([(margin_x + pad_x, div_y), (W - margin_x - pad_x, div_y)], fill=(*accent_rgb, 120), width=1)

        dr.text(((W - w_h) // 2, y0 + 14), headline, font=f_head, fill=(*accent_rgb, 255))
        dr.text(((W - w_a) // 2, y0 + 52), attrib, font=f_attr, fill=(236, 244, 252, 245))
        tx = margin_x + pad_x
        for li, ln in enumerate(lines):
            dr.text((tx, para_y0 + li * line_h), ln, font=f_para, fill=(230, 238, 247, 248))

    arr = np.array(img, dtype=np.float32)
    bgr = arr[:, :, [2, 1, 0]]
    alpha = arr[:, :, 3:4] / 255.0
    return bgr, alpha


def card_alpha_at(t, t_start, fade_in, hold_end, fade_out):
    if t < t_start or t > fade_out:
        return 0.0
    if t < fade_in:
        return float(R.smoothstep(t_start, fade_in, t))
    if t <= hold_end:
        return 1.0
    return float(1.0 - R.smoothstep(hold_end, fade_out, t))


def _map_colors_ramp(colors, warm, neut, cool):
    c = colors / 255.0
    mx = c.max(axis=1)
    mn = c.min(axis=1)
    lum = 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]

    warmish = (r > b + 0.06) & (r >= g - 0.02)
    coolish = (b >= r - 0.01) | (g > r + 0.05)

    pos = np.clip(np.power(np.clip(lum, 0.0, 1.0), 0.8) * 4.0, 0.0, 4.0)
    idx0 = np.clip(np.floor(pos).astype(int), 0, 3)
    frac = (pos - idx0)[:, None]

    out = np.zeros((len(c), 3), dtype=np.float32)
    for i in range(len(c)):
        i0 = idx0[i]
        w_col = warm[i0] * (1.0 - frac[i, 0]) + warm[i0 + 1] * frac[i, 0]
        n_col = neut[i0] * (1.0 - frac[i, 0]) + neut[i0 + 1] * frac[i, 0]
        c_col = cool[i0] * (1.0 - frac[i, 0]) + cool[i0 + 1] * frac[i, 0]

        if mx[i] < 0.20:
            col = c_col
        elif sat[i] < 0.18:
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


def _render_doorway_entry_frame(frame_rgb, tri, prev_cols, ramp_cfg, hatch_map):
    """Render doorway & elevator button-press frames using story-specific Cubist ramps."""
    cur_cols = CC.sample_triangle_colors(frame_rgb, tri)
    cols_sm = cur_cols if prev_cols is None else (0.72 * cur_cols + 0.28 * prev_cols)
    chosen = _map_colors_ramp(cols_sm, ramp_cfg["WARM"], ramp_cfg["NEUT"], ramp_cfg["COOL"])
    fill = np.clip(0.92 * chosen + 0.08 * cols_sm, 0, 255).astype(int)

    img = Image.new("RGB", (W, H), ramp_cfg["ink"])
    dr = ImageDraw.Draw(img)
    pts, simps = tri.points, tri.simplices
    ink_col = ramp_cfg["ink"]
    for s, c in zip(simps, fill):
        dr.polygon([tuple(pts[idx]) for idx in s], fill=tuple(c), outline=ink_col, width=2)
    out = np.asarray(img).astype(np.float32)

    lum_out = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    w_hatch = R.smoothstep(25.0, 60.0, lum_out) * (1.0 - R.smoothstep(185.0, 235.0, lum_out))
    out *= (1.0 + (0.065 * w_hatch * hatch_map)[:, :, None])

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dog = cv2.GaussianBlur(gray, (0, 0), 0.75) - cv2.GaussianBlur(gray, (0, 0), 2.0)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) * 0.25
    dark_gate = 1.0 - R.smoothstep(145.0, 210.0, gray)
    ink = np.clip((np.clip((grad - 18.0) / 55.0, 0.0, 1.0) * 0.78 + np.clip((-dog - 2.2) / 8.0, 0.0, 1.0) * 0.65) * dark_gate, 0.0, 0.84)[:, :, None]
    out = out * (1.0 - ink) + np.array(ink_col, dtype=np.float32) * ink
    return np.clip(out[:, :, [2, 1, 0]], 0, 255).astype(np.float32), cols_sm


def _prepare_protagonist_canvas(style_key):
    """Build hybrid high-res Cubist protagonist artwork + fine Delaunay shimmer map & bust mask."""
    path = os.path.join(PROTAG_DIR, f"{style_key}_elevator.png")
    rgb = np.array(Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS))
    bgr = rgb[:, :, [2, 1, 0]].astype(np.float32)

    # Fine Delaunay mesh to weave living 12-fps facet shimmer into the painted artwork
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tri = CC.make_points(gray, spacing=120, seed=19)
    tri_cols = CC.sample_triangle_colors(rgb, tri)
    facet_img = Image.new("RGB", (W, H), CUBIST_RAMPS[style_key]["ink"])
    dr = ImageDraw.Draw(facet_img)
    pts, simps = tri.points, tri.simplices
    ink_rgb = CUBIST_RAMPS[style_key]["ink"]
    for s, c in zip(simps, np.clip(tri_cols, 0, 255).astype(int)):
        dr.polygon([tuple(pts[idx]) for idx in s], fill=tuple(c), outline=ink_rgb, width=1)
    facet_bgr = np.asarray(facet_img)[:, :, [2, 1, 0]].astype(np.float32)

    # Blend 84% master painted Cubist canvas + 16% subtle facet geometry
    base_bgr = 0.84 * bgr + 0.16 * facet_bgr

    # Soft central bust mask for subtle 2-layer figure/elevator parallax breathing
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bust_d = np.hypot((xx - 0.50 * W) / (0.28 * W), (yy - 0.56 * H) / (0.42 * H))
    bust_mask = (1.0 - R.smoothstep(0.75, 1.15, bust_d))[:, :, None]

    # Precompute ceiling shimmer / light-map mask
    ceil_w = (1.0 - R.smoothstep(0.28 * H, 0.55 * H, yy)) * R.smoothstep(95.0, 210.0, gray)
    vig_rx = (xx - 0.5 * W) / (0.56 * W)
    vig_ry = (yy - 0.5 * H) / (0.56 * H)
    vignette = np.clip(1.0 - 0.22 * (vig_rx * vig_rx + vig_ry * vig_ry), 0.72, 1.0)[:, :, None]

    # Contours & hatching weights
    dog = cv2.GaussianBlur(gray, (0, 0), 0.75) - cv2.GaussianBlur(gray, (0, 0), 1.9)
    ink_contour = np.clip((-dog - 3.5) / 14.0, 0.0, 0.35)[:, :, None]
    w_hatch = (R.smoothstep(24.0, 62.0, gray) * (1.0 - R.smoothstep(185.0, 240.0, gray))).astype(np.float32)

    return dict(
        base_bgr=base_bgr,
        bust_mask=bust_mask,
        ceil_w=ceil_w,
        vignette=vignette,
        ink_contour=ink_contour,
        w_hatch=w_hatch,
        yy=yy,
        xx=xx,
    )


def render_protagonist_pose(canvas, style_key, i_frame, hatch_map):
    """Render one animated 12-fps frame of the protagonist inside the narrative elevator."""
    t_tail = i_frame / 24.0
    u = float(np.clip((i_frame - 54.0) / (515.0 - 54.0), 0.0, 1.0))

    # Gentle camera push-in (1.00 -> 1.055) anchored at upper-center face pivot (0.50W, 0.36H)
    zoom = 1.00 + 0.055 * (u * u * (3.0 - 2.0 * u))
    sway_x = 3.2 * np.sin(2.0 * np.pi * t_tail / 8.0)
    sway_y = 2.4 * np.cos(2.0 * np.pi * t_tail / 6.5)

    px, py = 0.50 * W, 0.36 * H
    M_bg = np.array([
        [zoom, 0.0, px * (1.0 - zoom) + sway_x],
        [0.0, zoom, py * (1.0 - zoom) + sway_y],
    ], dtype=np.float32)
    M_fg = np.array([
        [zoom * 1.008, 0.0, px * (1.0 - zoom * 1.008) + sway_x * 1.35],
        [0.0, zoom * 1.008, py * (1.0 - zoom * 1.008) + sway_y * 1.35 + 1.5 * np.sin(2.0 * np.pi * t_tail / 4.2)],
    ], dtype=np.float32)

    bg_w = cv2.warpAffine(canvas["base_bgr"], M_bg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    fg_w = cv2.warpAffine(canvas["base_bgr"], M_fg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    bm_w = cv2.warpAffine(canvas["bust_mask"][:, :, 0], M_fg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[:, :, None]
    out = bg_w * (1.0 - bm_w) + fg_w * bm_w

    # Story-specific 12-fps atmospheric elevator lighting
    if style_key == "nansen":
        # Aurora borealis shimmer drifting across ceiling ice panes + frosted vignette
        wave = 0.5 + 0.5 * np.sin(0.012 * canvas["xx"] - 0.008 * canvas["yy"] + 2.0 * np.pi * t_tail / 4.5)
        aur = (canvas["ceil_w"] * (0.45 + 0.55 * wave))[:, :, None]
        out += aur * np.array([28.0, 34.0, 10.0], dtype=np.float32) * 0.65
        out *= canvas["vignette"]
    elif style_key == "shackleton":
        # Warm oil-lantern flicker in ceiling bulkhead + rich vignette
        flicker = 0.65 + 0.35 * np.sin(2.0 * np.pi * t_tail / 2.9 + 0.7 * np.sin(2.0 * np.pi * t_tail / 1.3))
        lantern = (canvas["ceil_w"] * flicker)[:, :, None]
        out += lantern * np.array([10.0, 24.0, 38.0], dtype=np.float32) * 0.60
        out *= canvas["vignette"]
    else:
        # Mars500: cool circadian airlock ambiance -> luminous hatch-open warm gold across finale
        warm_mix = float(R.smoothstep(3.0, 18.0, t_tail))
        cool_pulse = (1.0 - warm_mix) * canvas["ceil_w"][:, :, None] * np.array([26.0, 24.0, 6.0], dtype=np.float32) * 0.45
        warm_glow = warm_mix * (0.35 + 0.65 * canvas["ceil_w"])[:, :, None] * np.array([-8.0, 10.0, 24.0], dtype=np.float32) * 0.60
        out = out + cool_pulse + warm_glow
        out *= canvas["vignette"]

    # Subtle 12-fps chalk paper-tooth + charcoal outline crispness
    out *= (1.0 + (0.045 * canvas["w_hatch"] * hatch_map)[:, :, None])
    out *= (1.0 - canvas["ink_contour"] * 0.45)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def render_single_tail(vcfg, out_tail):
    key = vcfg["key"]
    style = vcfg["style"]
    ramp_cfg = CUBIST_RAMPS[style]
    outro_bgr, outro_a = render_story_card(vcfg["outro"], vcfg["accent"], is_intro=False)

    raw_frames = CC.read_frames_24fps(TAIL_RAW, W, H)  # RGB 516 frames
    n_frames = len(raw_frames)
    acc = sum(raw_frames[k].mean(axis=2) for k in range(0, min(84, n_frames), 12)) / max(1, len(range(0, min(84, n_frames), 12)))
    tri_door = CC.make_points(acc)

    canvas = _prepare_protagonist_canvas(style)
    hatch_bank = R.build_hatching_bank(12, H, W, seed=99)
    rng = np.random.default_rng(77 + sum(ord(c) for c in key))

    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "14", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", "24", out_tail, "-y",
        ],
        stdin=subprocess.PIPE,
    )

    prev_cols = None
    cached_door_bgr = None
    cached_prot_bgr = None
    for i in range(n_frames):
        pose_idx = i // 2
        hatch = hatch_bank[pose_idx % len(hatch_bank)]

        if i <= 80 and (i % 2 == 0 or cached_door_bgr is None):
            cached_door_bgr, prev_cols = _render_doorway_entry_frame(raw_frames[i], tri_door, prev_cols, ramp_cfg, hatch)

        if i >= 48 and (i % 2 == 0 or cached_prot_bgr is None):
            cached_prot_bgr = render_protagonist_pose(canvas, style, i, hatch)

        if i < 50:
            styled = cached_door_bgr.copy()
        elif i <= 78:
            mix = float(R.smoothstep(50.0, 78.0, i))
            styled = cached_door_bgr * (1.0 - mix) + cached_prot_bgr * mix
        else:
            styled = cached_prot_bgr.copy()

        t_tail = i / 24.0
        ca = card_alpha_at(t_tail, 4.2, 5.6, 17.5, 19.2)
        if ca > 0.001:
            a_eff = outro_a * ca
            styled = styled * (1.0 - a_eff) + outro_bgr * a_eff

        styled += rng.normal(0.0, 1.6, (H, W, 1)).astype(np.float32)
        enc.stdin.write(np.clip(styled, 0, 255).astype(np.uint8).tobytes())

    enc.stdin.close()
    enc.wait()
    print(f"Protagonist Cubist tail ({style}): {n_frames} frames -> {out_tail}")


def render_all_tails(videos):
    os.makedirs(TRI_DIR, exist_ok=True)
    for vcfg in videos:
        tail_p = os.path.join(TRI_DIR, f"tail_{vcfg['key']}.mp4")
        render_single_tail(vcfg, tail_p)


if __name__ == "__main__":
    from pipeline import build_trilogy_films as BTF
    render_all_tails(BTF.VIDEOS)
