#!/usr/bin/env python3
"""Prepare mastered archival voiceover quotes and 12-fps stylized archival screen MP4s
for the 4-Act 'Into the Unknown' Night-Chalk Rotoscope film (Nansen / Shackleton / ESA Mars500).
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import media_base as B

ASSETS_DIR = os.path.join(B.WORK, "expedition_assets")
RAW_DIR = os.path.join(ASSETS_DIR, "raw")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
VIDEO_DIR = os.path.join(ASSETS_DIR, "video")
DIAG_DIR = os.path.join(B.WORK, "track_diag")

W_SCR, H_SCR = 720, 450
HUD_H = 38
CAP_H = 72
BODY_H = H_SCR - HUD_H - CAP_H


def ensure_dirs():
    for d in (RAW_DIR, AUDIO_DIR, VIDEO_DIR, DIAG_DIR):
        os.makedirs(d, exist_ok=True)


def run_ff(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


def wav_dur(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip())


def extract_clean_slice(src_mp3, start_s, dur_s, out_wav, vintage_1910=False):
    eq = (
        "highpass=f=160,lowpass=f=4200,afftdn=nf=-25,"
        "acompressor=threshold=-18dB:ratio=3.0:attack=15:release=180,"
        "loudnorm=I=-15:TP=-1.5:LRA=9"
        if vintage_1910
        else "highpass=f=85,lowpass=f=11000,"
        "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180,"
        "loudnorm=I=-15:TP=-1.5:LRA=9"
    )
    run_ff([
        "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", src_mp3,
        "-af", f"{eq},afade=t=in:st=0:d=0.06,afade=t=out:st={max(0.05, dur_s - 0.08):.3f}:d=0.08",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", out_wav, "-y",
    ])


def concat_with_silences(parts_and_gaps, out_wav):
    """parts_and_gaps: list of (wav_path, trailing_silence_s)."""
    tmp_files = []
    lst_path = out_wav + ".txt"
    sil_idx = 0
    with open(lst_path, "w") as lf:
        for wav_p, gap_s in parts_and_gaps:
            lf.write(f"file '{os.path.abspath(wav_p)}'\n")
            if gap_s > 0:
                sp = f"{out_wav}.sil{sil_idx}.wav"
                sil_idx += 1
                run_ff([
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-t", f"{gap_s:.3f}", "-c:a", "pcm_s16le", sp, "-y",
                ])
                tmp_files.append(sp)
                lf.write(f"file '{os.path.abspath(sp)}'\n")
    raw_cat = out_wav + ".cat.wav"
    run_ff(["-f", "concat", "-safe", "0", "-i", lst_path, "-c", "copy", raw_cat, "-y"])
    dur = wav_dur(raw_cat)
    run_ff([
        "-i", raw_cat,
        "-af", f"loudnorm=I=-15:TP=-1.5:LRA=9,afade=t=in:st=0:d=0.28,afade=t=out:st={max(0.1, dur - 0.28):.3f}:d=0.28",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", out_wav, "-y",
    ])
    for f in [lst_path, raw_cat, *tmp_files]:
        if os.path.exists(f):
            os.remove(f)


def build_all_quotes():
    tmp = os.path.join(AUDIO_DIR, "_parts")
    os.makedirs(tmp, exist_ok=True)

    # Quote 1: Nansen Sec 06 Departure -> "...into the unknown, where we two alone and the dogs were to wander for months."
    q1a, q1b = os.path.join(tmp, "q1a.wav"), os.path.join(tmp, "q1b.wav")
    extract_clean_slice(os.path.join(RAW_DIR, "nansen_06.mp3"), 283.40, 296.22 - 283.40, q1a)
    extract_clean_slice(os.path.join(RAW_DIR, "nansen_06.mp3"), 301.48, 318.20 - 301.48, q1b)
    concat_with_silences([(q1a, 0.42), (q1b, 0.0)], os.path.join(AUDIO_DIR, "quote1.wav"))

    # Quote 2: Shackleton 1910 Phonograph -> "...the lack of 50 pounds of food."
    q2 = os.path.join(tmp, "q2.wav")
    extract_clean_slice(os.path.join(RAW_DIR, "shackleton_1910_voice.mp3"), 0.50, 24.40, q2, vintage_1910=True)
    concat_with_silences([(q2, 0.0)], os.path.join(AUDIO_DIR, "quote2.wav"))

    # Quote 3: Shackleton South! Ch. IV -> "We were helpless intruders in a strange world..."
    q3 = os.path.join(tmp, "q3.wav")
    extract_clean_slice(os.path.join(RAW_DIR, "shackleton_07.mp3"), 1722.15, 23.85, q3)
    concat_with_silences([(q3, 0.0)], os.path.join(AUDIO_DIR, "quote3.wav"))

    # Quote 4: ESA Mars500 Diego Urbina -> "My name is Diego Urbina... smiling and laughing, it becomes quite important."
    q4 = os.path.join(tmp, "q4.wav")
    extract_clean_slice(os.path.join(RAW_DIR, "mars500_diego.mp3"), 0.00, 25.90, q4)
    concat_with_silences([(q4, 0.0)], os.path.join(AUDIO_DIR, "quote4.wav"))

    # Quote 5: ESA Mars500 Romain Charles -> "When the hatch opens on the 4th of November, I'm looking forward to seeing my family... I would very much like it."
    q5a, q5b = os.path.join(tmp, "q5a.wav"), os.path.join(tmp, "q5b.wav")
    extract_clean_slice(os.path.join(RAW_DIR, "mars500_romain_forward.mp3"), 9.60, 6.75, q5a)
    extract_clean_slice(os.path.join(RAW_DIR, "mars500_romain_forward.mp3"), 23.72, 10.40, q5b)
    concat_with_silences([(q5a, 0.35), (q5b, 0.0)], os.path.join(AUDIO_DIR, "quote5.wav"))

    # Quote 6: Nansen Land, Land! + 2.0s breathing gap across 266.5s Cubist cut + Shackleton "form a nucleus for the next expedition"
    q6a, q6b, q6c = os.path.join(tmp, "q6a.wav"), os.path.join(tmp, "q6b.wav"), os.path.join(tmp, "q6c.wav")
    extract_clean_slice(os.path.join(RAW_DIR, "nansen_11.mp3"), 20.40, 35.40 - 20.40, q6a)
    extract_clean_slice(os.path.join(RAW_DIR, "nansen_11.mp3"), 55.28, 59.95 - 55.28, q6b)
    extract_clean_slice(os.path.join(RAW_DIR, "shackleton_32.mp3"), 445.15, 458.20 - 445.15, q6c)
    concat_with_silences([(q6a, 0.35), (q6b, 2.00), (q6c, 0.0)], os.path.join(AUDIO_DIR, "quote6.wav"))

    for k in range(1, 7):
        p = os.path.join(AUDIO_DIR, f"quote{k}.wav")
        print(f"quote{k}.wav: {wav_dur(p):.2f}s")


def load_font(size):
    for fp in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


FONT_HUD = load_font(17)
FONT_CAP = load_font(22)


def stylize_archival_frame(bgr, palette_tint, scanline_phase=0):
    """Transform archival video/still body frame into high-contrast Night-Chalk style."""
    h, w = bgr.shape[:2]
    sm = cv2.bilateralFilter(bgr, 7, 42, 8)
    sm = cv2.bilateralFilter(sm, 5, 34, 6)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2Lab)
    L_u8, a_u8, b_u8 = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L_u8).astype(np.float32)

    # Quantized sculptural chalk bands
    step = 28.0
    q_idx = np.floor(L / step)
    phase = (L / step) - q_idx - 0.5
    L_q = (q_idx + 0.5 + 0.5 * np.tanh(phase * 4.0) / np.tanh(2.0)) * step
    L_out = np.clip(0.35 * L + 0.65 * L_q, 10.0, 248.0)

    a_off = a_u8.astype(np.float32) - 128.0
    b_off = b_u8.astype(np.float32) - 128.0
    a_shift, b_shift, color_mix = palette_tint
    lit_w = np.clip((L_out - 24.0) / 140.0, 0.0, 1.0)
    a_new = np.clip(128.0 + a_off * color_mix + a_shift * lit_w, 0.0, 255.0)
    b_new = np.clip(128.0 + b_off * color_mix + b_shift * lit_w, 0.0, 255.0)

    out = cv2.cvtColor(cv2.merge([L_out.astype(np.uint8), a_new.astype(np.uint8), b_new.astype(np.uint8)]), cv2.COLOR_Lab2BGR).astype(np.float32)

    # Charcoal contours
    sx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy) * 0.24
    ink = np.clip((grad - 12.0) / 38.0, 0.0, 0.85)
    out *= (1.0 - ink[:, :, None])

    # Subtle scanline texture for dashboard monitor realism
    yy = (np.arange(h) + scanline_phase) % 4
    scan = np.where(yy == 0, 0.90, 1.02).astype(np.float32)[:, None, None]
    out *= scan
    return np.clip(out, 0, 255).astype(np.uint8)


def compose_hud_frame(body_bgr, hud_text, caption_text, accent_rgb):
    canvas = Image.new("RGB", (W_SCR, H_SCR), (14, 16, 20))
    body_rgb = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(body_rgb), (0, HUD_H))
    dr = ImageDraw.Draw(canvas)

    # Top telemetry bar
    dr.rectangle([0, 0, W_SCR, HUD_H], fill=(12, 16, 22))
    dr.line([(0, HUD_H - 1), (W_SCR, HUD_H - 1)], fill=accent_rgb, width=2)
    dr.ellipse([16, 12, 28, 24], fill=accent_rgb)
    dr.text((38, 10), hud_text, font=FONT_HUD, fill=(235, 242, 248))

    # Bottom high-contrast caption banner
    dr.rectangle([0, H_SCR - CAP_H, W_SCR, H_SCR], fill=(10, 13, 18))
    dr.line([(0, H_SCR - CAP_H), (W_SCR, H_SCR - CAP_H)], fill=accent_rgb, width=2)
    tb = dr.textbbox((0, 0), caption_text, font=FONT_CAP)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    dr.text(((W_SCR - tw) // 2, H_SCR - CAP_H + (CAP_H - th) // 2 - 2), caption_text, font=FONT_CAP, fill=(248, 250, 252))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def crop_cover(fr, src_crop, out_w=W_SCR, out_h=BODY_H):
    y0, y1, x0, x1 = src_crop
    sub = fr[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    scale = max(out_w / sw, out_h / sh)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    rs = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    cx = (nw - out_w) // 2
    cy = (nh - out_h) // 2
    return rs[cy : cy + out_h, cx : cx + out_w]


def extract_video_frames(path, start_s, dur_s=3.0, n_frames=72):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    for k in range(n_frames):
        src_idx = int(round((start_s + k / 24.0) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ok, fr = cap.read()
        if not ok and frames:
            fr = frames[-1].copy()
        frames.append(fr)
    cap.release()
    return frames


def ken_burns_frames(img_path, zoom_start=1.0, zoom_end=1.12, pan_x=(0.5, 0.5), pan_y=(0.5, 0.5), n_frames=72):
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
        crop = im[y0 : y0 + ch, x0 : x0 + cw]
        frames.append(cv2.resize(crop, (W_SCR, BODY_H), interpolation=cv2.INTER_CUBIC))
    return frames


def write_screen_mp4(frames_bgr, out_path):
    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W_SCR}x{H_SCR}", "-r", "24", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", "24", out_path, "-y",
        ],
        stdin=subprocess.PIPE,
    )
    for fr in frames_bgr:
        enc.stdin.write(fr.tobytes())
    enc.stdin.close()
    enc.wait()


def build_all_screens():
    specs = [
        {
            "panel": 1,
            "kind": "video",
            "src": os.path.join(RAW_DIR, "ice_breaker.mp4"),
            "start": 125.5,
            "crop": (0, 310, 0, 640),  # excludes bottom Periscope bar
            "tint": (-7.0, -21.0, 0.40),  # cool ice-blue
            "accent": (110, 215, 255),
            "hud": "FRAM EXPEDITION • 84°04'N • MARCH 1895",
            "cap": "INTO THE GREAT SILENT WORLD OF ICE",
        },
        {
            "panel": 2,
            "kind": "video",
            "src": os.path.join(RAW_DIR, "shackleton_ship.mp4"),
            "start": 32.5,
            "crop": (45, 675, 60, 1080),  # excludes top-right Telegraph diamond logo; 32.5-35.5s stern/helm has zero captions
            "tint": (11.0, 24.0, 0.90),  # warm amber/ochre
            "accent": (255, 195, 95),
            "hud": "IMPERIAL TRANS-ANTARCTIC • 69°11'S • 1915",
            "cap": "ENDURANCE BESET IN THE WEDDELL PACK",
        },
        {
            "panel": 3,
            "kind": "video",
            "src": os.path.join(RAW_DIR, "ice_breaker.mp4"),
            "start": 483.5,
            "crop": (0, 310, 0, 640),
            "tint": (12.0, 22.0, 0.60),  # high-contrast ember/amber
            "accent": (255, 165, 85),
            "hud": "PATIENCE CAMP & SLEDGE MARCH • 1916",
            "cap": "INTRUDERS IN A STRANGE ELEMENTAL WORLD",
        },
        {
            "panel": 4,
            "kind": "image",
            "src": os.path.join(RAW_DIR, "mars500_crew.webp"),
            "kb": (1.02, 1.14, (0.42, 0.52), (0.32, 0.25)),
            "tint": (-14.0, -9.0, 1.15),  # electric teal blueprint
            "accent": (85, 245, 230),
            "hud": "ESA MARS500 • MOSCOW IBMP • DAY 260 / 520",
            "cap": "SIX CREWMATES SEALED IN STEEL ISOLATION",
        },
        {
            "panel": 5,
            "kind": "image",
            "src": os.path.join(RAW_DIR, "mars500_bluelight.webp"),
            "kb": (1.02, 1.16, (0.60, 0.72), (0.45, 0.40)),
            "tint": (-6.0, -24.0, 1.15),  # deep cyan blue chamber
            "accent": (95, 220, 255),
            "hud": "MARS500 HABITAT • CIRCADIAN SOL 500",
            "cap": "520 DAYS WITHOUT EARTH SUNLIGHT",
        },
        {
            "panel": 6,
            "kind": "image",
            "src": os.path.join(RAW_DIR, "mars500_diego_painting.webp"),
            "kb": (1.00, 1.10, (0.55, 0.50), (0.35, 0.42)),
            "tint": (7.0, 16.0, 1.30),  # Diego's own chalk pastel Mars painting pushed to luminous gold
            "accent": (255, 220, 115),
            "hud": "NOV 4, 2011 • HATCH UNSEALED • NEXT FRONTIER",
            "cap": "LAND AT LAST — TOWARD THE NEXT HORIZON",
        },
    ]

    mid_thumbs = []
    for sp in specs:
        p = sp["panel"]
        if sp["kind"] == "video":
            raw_frames = extract_video_frames(sp["src"], sp["start"], 3.0, 72)
            body_frames = [crop_cover(fr, sp["crop"]) for fr in raw_frames]
        else:
            z0, z1, px, py = sp["kb"]
            body_frames = ken_burns_frames(sp["src"], z0, z1, px, py, 72)

        # 12-fps cadence lock to integrate cleanly with 12-fps chalk rotoscope car interior
        out_frames = []
        cached_pose = None
        for idx, bfr in enumerate(body_frames):
            if idx % 2 == 0 or cached_pose is None:
                styl = stylize_archival_frame(bfr, sp["tint"], scanline_phase=(idx // 2) % 4)
                cached_pose = compose_hud_frame(styl, sp["hud"], sp["cap"], sp["accent"])
            out_frames.append(cached_pose)

        out_mp4 = os.path.join(VIDEO_DIR, f"screen{p}.mp4")
        write_screen_mp4(out_frames, out_mp4)
        print(f"screen{p}.mp4: 72 frames -> {out_mp4}")

        thumb = cv2.resize(out_frames[36], (360, 225))
        cv2.putText(thumb, f"P{p}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 180), 2)
        mid_thumbs.append(thumb)

    sheet = np.vstack([np.hstack(mid_thumbs[:3]), np.hstack(mid_thumbs[3:])])
    sheet_path = os.path.join(DIAG_DIR, "expedition_screens_sheet.jpg")
    cv2.imwrite(sheet_path, sheet)
    print("Wrote", sheet_path)


if __name__ == "__main__":
    ensure_dirs()
    build_all_quotes()
    build_all_screens()
