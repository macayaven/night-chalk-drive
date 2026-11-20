#!/usr/bin/env python3
"""Generate 18 high-legibility progressive dashboard screen MP4s (720x450, 24fps, 72f)
for the 3 trilogy films (Nansen, Shackleton, ESA Mars500) + 3x6 inspection grid.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import fetch_archival_media as FEA, media_base as B

RAW_DIR = FEA.RAW_DIR
SCR_DIR = os.path.join(B.WORK, "trilogy_assets", "screens")
DIAG_DIR = FEA.DIAG_DIR

W_SCR, H_SCR = 720, 450
HUD_H = 54
CAP_H = 86
BODY_H = H_SCR - HUD_H - CAP_H  # 310

FONT_STAGE = FEA.load_font(25)
FONT_TITLE = FEA.load_font(33)


def stylize_trilogy_screen_body(bgr, tint, lift_contrast=True):
    """High-visibility stylization tuned for ~130x85 dashboard screen projection."""
    sm = cv2.bilateralFilter(bgr, 7, 45, 8)
    sm = cv2.bilateralFilter(sm, 5, 36, 6)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2Lab)
    L_u8, a_u8, b_u8 = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(L_u8).astype(np.float32)
    if lift_contrast:
        # Lift shadows into 50..248 so dashboard details remain bold under ambient car interior glow
        L = np.clip((L - 10.0) * 1.12 + 42.0, 28.0, 250.0)

    step = 28.0
    q_idx = np.floor(L / step)
    phase = (L / step) - q_idx - 0.5
    L_q = (q_idx + 0.5 + 0.5 * np.tanh(phase * 4.2) / np.tanh(2.1)) * step
    L_out = np.clip(0.32 * L + 0.68 * L_q, 24.0, 250.0)

    a_off = a_u8.astype(np.float32) - 128.0
    b_off = b_u8.astype(np.float32) - 128.0
    a_shift, b_shift, color_mix = tint
    lit_w = np.clip((L_out - 28.0) / 145.0, 0.0, 1.0)
    a_new = np.clip(128.0 + a_off * color_mix + a_shift * lit_w, 0.0, 255.0)
    b_new = np.clip(128.0 + b_off * color_mix + b_shift * lit_w, 0.0, 255.0)

    out = cv2.cvtColor(cv2.merge([L_out.astype(np.uint8), a_new.astype(np.uint8), b_new.astype(np.uint8)]), cv2.COLOR_Lab2BGR).astype(np.float32)

    sx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy) * 0.26
    ink = np.clip((grad - 11.0) / 34.0, 0.0, 0.88)
    out *= (1.0 - ink[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def compose_trilogy_screen_frame(body_bgr, step_idx, total_steps, stage_label, title_text, accent_rgb):
    canvas = Image.new("RGB", (W_SCR, H_SCR), (10, 13, 18))
    canvas.paste(Image.fromarray(cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB)), (0, HUD_H))
    dr = ImageDraw.Draw(canvas)

    # Top progress strip (1..6) + bold stage badge
    dr.rectangle([0, 0, W_SCR, HUD_H], fill=(10, 14, 22))
    seg_w = W_SCR / total_steps
    for s in range(total_steps):
        x0 = int(round(s * seg_w)) + 4
        x1 = int(round((s + 1) * seg_w)) - 4
        col = accent_rgb if s < step_idx else (42, 50, 62)
        dr.rectangle([x0, 4, x1, 11], fill=col)

    dr.line([(0, HUD_H - 1), (W_SCR, HUD_H - 1)], fill=accent_rgb, width=3)
    dr.ellipse([18, 22, 34, 38], fill=accent_rgb)
    dr.text((46, 17), stage_label, font=FONT_STAGE, fill=(242, 248, 255))

    # High-contrast bottom title banner
    dr.rectangle([0, H_SCR - CAP_H, W_SCR, H_SCR], fill=(8, 10, 15))
    dr.line([(0, H_SCR - CAP_H), (W_SCR, H_SCR - CAP_H)], fill=accent_rgb, width=3)
    tb = dr.textbbox((0, 0), title_text, font=FONT_TITLE)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (W_SCR - tw) // 2
    ty = H_SCR - CAP_H + (CAP_H - th) // 2 - 4
    dr.text((tx + 2, ty + 2), title_text, font=FONT_TITLE, fill=(0, 0, 0))
    dr.text((tx, ty), title_text, font=FONT_TITLE, fill=accent_rgb if step_idx == total_steps else (250, 252, 255))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def ken_burns_body(img_path, zoom_start, zoom_end, pan_x, pan_y, n_frames=72):
    im = cv2.imread(img_path)
    ih, iw = im.shape[:2]
    frames = []
    for k in range(n_frames):
        u = k / max(1, n_frames - 1)
        z = zoom_start + (zoom_end - zoom_start) * u
        px = pan_x[0] + (pan_x[1] - pan_x[0]) * u
        py = pan_y[0] + (pan_y[1] - pan_y[0]) * u
        cw = int(round(iw / z))
        ch = int(round(cw * (BODY_H / W_SCR)))
        if ch > ih:
            ch = ih
            cw = int(round(ch * (W_SCR / BODY_H)))
        x0 = int(np.clip(round(px * (iw - cw)), 0, iw - cw))
        y0 = int(np.clip(round(py * (ih - ch)), 0, ih - ch))
        frames.append(cv2.resize(im[y0 : y0 + ch, x0 : x0 + cw], (W_SCR, BODY_H), interpolation=cv2.INTER_CUBIC))
    return frames


def all_specs():
    ib = os.path.join(RAW_DIR, "ice_breaker.mp4")
    sh = os.path.join(RAW_DIR, "shackleton_ship.mp4")
    m_crew = os.path.join(RAW_DIR, "mars500_crew.webp")
    m_blue = os.path.join(RAW_DIR, "mars500_bluelight.webp")
    m_diego = os.path.join(RAW_DIR, "mars500_diego_painting.webp")

    return {
        "nansen": [
            (1, "video", ib, 125.5, (0, 310, 0, 640), (-8.5, -22.0, 0.35), (120, 230, 255), "1/6 • 78°50'N • FRAM", "FRAM IN PACK ICE"),
            (2, "video", ib, 483.5, (0, 310, 0, 640), (-8.0, -21.0, 0.35), (120, 230, 255), "2/6 • 84°04'N • SLEDGE", "INTO THE SOLITUDE"),
            (3, "video", ib, 636.5, (0, 310, 0, 640), (-9.0, -23.0, 0.30), (135, 238, 255), "3/6 • 85°30'N • DRIFT", "ENDLESS WHITE ICE"),
            (4, "video", ib, 534.5, (0, 310, 0, 640), (-7.5, -20.0, 0.35), (150, 242, 255), "4/6 • 86°14'N • APEX", "86°14'N FARTHEST"),
            (5, "video", ib, 789.5, (0, 310, 0, 640), (-5.0, -14.0, 0.45), (195, 248, 255), "5/6 • 81°30'N • LAND", "LAND AT LAST!"),
            (6, "video", ib, 738.5, (0, 310, 0, 640), (-4.0, -10.0, 0.55), (225, 250, 255), "6/6 • NORWAY • 1896", "RETURN TO LIGHT"),
        ],
        "shackleton": [
            (1, "video", sh, 8.2, (45, 455, 60, 1100), (11.5, 25.0, 0.85), (255, 195, 95), "1/6 • WEDDELL SEA • 1915", "BESET IN THE ICE"),
            (2, "video", sh, 49.8, (45, 470, 60, 1080), (13.0, 26.0, 0.85), (255, 175, 80), "2/6 • OCT 1915 • PRESSURE", "MILLION-TON CRUSH"),
            (3, "video", sh, 32.5, (45, 675, 60, 1080), (12.0, 24.0, 0.85), (255, 160, 75), "3/6 • NOV 21, 1915", "SHIP GONE BENEATH"),
            (4, "video", ib, 687.5, (0, 310, 0, 640), (10.5, 22.0, 0.50), (255, 195, 105), "4/6 • PATIENCE CAMP", "FIGHTING DESPAIR"),
            (5, "video", ib, 126.5, (0, 310, 0, 640), (11.0, 23.0, 0.50), (255, 210, 120), "5/6 • 800-MILE VOYAGE", "OPEN BOAT RESCUE"),
            (6, "video", sh, 51.8, (45, 470, 60, 1100), (9.5, 22.0, 0.95), (255, 225, 130), "6/6 • AUG 1916 • SAVED", "ALL 28 MEN ALIVE"),
        ],
        "mars500": [
            (1, "image", m_blue, (1.80, 2.02, (0.84, 0.86), (0.50, 0.50)), None, (-12.0, -18.0, 1.15), (90, 245, 235), "1/6 • SOL 001 • MOSCOW", "HATCH SEALED: 520d"),
            (2, "image", m_crew, (1.02, 1.12, (0.45, 0.52), (0.34, 0.28)), None, (-14.0, -10.0, 1.15), (90, 245, 235), "2/6 • SOL 105 • COMMS", "20-MIN SIGNAL DELAY"),
            (3, "image", m_blue, (1.00, 1.14, (0.35, 0.48), (0.52, 0.44)), None, (-8.0, -24.0, 1.15), (105, 235, 255), "3/6 • SOL 260 • MODULE", "NO EARTH SUNLIGHT"),
            (4, "image", m_diego, (1.50, 1.68, (0.72, 0.70), (0.18, 0.22)), None, (-4.0, 2.0, 1.25), (140, 245, 220), "4/6 • SOL 380 • EVA SIM", "MARS SURFACE WALK"),
            (5, "image", m_crew, (1.22, 1.36, (0.64, 0.54), (0.32, 0.26)), None, (2.0, 10.0, 1.25), (245, 225, 145), "5/6 • SOL 500 • RETURN", "HOMEWARD BOUND"),
            (6, "image", m_diego, (1.00, 1.10, (0.52, 0.48), (0.38, 0.44)), None, (8.0, 20.0, 1.30), (255, 225, 115), "6/6 • NOV 4, 2011 • SOL 520", "WELCOME TO EARTH"),
        ],
    }


def main():
    os.makedirs(SCR_DIR, exist_ok=True)
    os.makedirs(DIAG_DIR, exist_ok=True)
    specs = all_specs()
    rows = []
    for story in ("nansen", "shackleton", "mars500"):
        row_thumbs = []
        for idx, item in enumerate(specs[story], 1):
            p, kind, src, param, crop_rect, tint, accent, stage, title = item
            if kind == "video":
                raw = FEA.extract_video_frames(src, param, 3.0, 72)
                bodies = [FEA.crop_cover(fr, crop_rect, W_SCR, BODY_H) for fr in raw]
            else:
                z0, z1, px, py = param
                bodies = ken_burns_body(src, z0, z1, px, py, 72)

            out_frames = []
            cached = None
            for fi, bfr in enumerate(bodies):
                if fi % 2 == 0 or cached is None:
                    styl = stylize_trilogy_screen_body(bfr, tint)
                    cached = compose_trilogy_screen_frame(styl, p, 6, stage, title, accent)
                out_frames.append(cached)

            out_mp4 = os.path.join(SCR_DIR, f"{story}_screen{p}.mp4")
            FEA.write_screen_mp4(out_frames, out_mp4)
            th = cv2.resize(out_frames[36], (320, 200))
            row_thumbs.append(th)
            print(f"{story}_screen{p}.mp4 -> 72 frames")
        rows.append(np.hstack(row_thumbs))

    sheet = np.vstack(rows)
    out_sheet = os.path.join(DIAG_DIR, "trilogy_screens_sheet.jpg")
    cv2.imwrite(out_sheet, sheet)
    print("Wrote", out_sheet)


if __name__ == "__main__":
    main()
