#!/usr/bin/env python3
"""Build visual assets and the 15s 2x2 debug clip for the technical note and social cards.

Outputs inside ``--site-note-dir`` (and ``<data-dir>/output`` for LinkedIn):
  1. ``<data-dir>/output/linkedin_debug_2x2_15s.mp4`` (1080x1440, 24fps, 15s)
  2. ``<site-note-dir>/debug-2x2.mp4`` (<8 MB web encode)
  3. ``<site-note-dir>/tracker-comparison.jpg`` (1400x760)
  4. ``<site-note-dir>/trilogy-palettes.jpg`` (1984x392)
  5. ``<site-note-dir>/og.png`` (1200x627)
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths

ROOT = str(paths.MEDIA_ROOT)
SITE_NOTE_DIR = str(paths.DOCS_IMAGES_DIR)
OUT_LINKEDIN = ""
OUT_WEB = ""
STORY288 = ""
DEFECTIVE = ""
NANSEN = ""
SHACKLETON = ""
MARS500 = ""
AUDIO_WAV = ""
QUADS_PATH = ""
ERR_SEG1 = ""
ERR_SEG4 = ""


def configure_paths(data_dir=None, site_note_dir=None):
    """Resolve input media paths and output directories from CLI flags or environment."""
    global ROOT, SITE_NOTE_DIR, OUT_LINKEDIN, OUT_WEB
    global STORY288, DEFECTIVE, NANSEN, SHACKLETON, MARS500
    global AUDIO_WAV, QUADS_PATH, ERR_SEG1, ERR_SEG4

    ROOT = str(Path(data_dir or paths.MEDIA_ROOT).expanduser().resolve())
    SITE_NOTE_DIR = str(Path(site_note_dir or paths.DOCS_IMAGES_DIR).expanduser().resolve())
    OUT_LINKEDIN = os.path.join(ROOT, "output", "linkedin_debug_2x2_15s.mp4")
    OUT_WEB = os.path.join(SITE_NOTE_DIR, "debug-2x2.mp4")

    STORY288 = os.path.join(ROOT, "work", "story288.mp4")
    DEFECTIVE = os.path.join(ROOT, "output", "night-drive_colour-cartoon_forward_defective_prev.mp4")
    NANSEN = os.path.join(ROOT, "output", "01_nansen_into-the-white-silence_themed-ending.mp4")
    SHACKLETON = os.path.join(ROOT, "output", "02_shackleton_endurance-in-the-dark_themed-ending.mp4")
    MARS500 = os.path.join(ROOT, "output", "03_mars500_520-days-to-another-world_themed-ending.mp4")
    AUDIO_WAV = os.path.join(ROOT, "work", "audio288", "trilogy_01_nansen_master.wav")

    work_quads = os.path.join(ROOT, "work", "track_quads_full_drive.json")
    QUADS_PATH = work_quads if os.path.isfile(work_quads) else paths.QUADS_FULL_DRIVE

    ERR_SEG1 = os.path.join(ROOT, "docs", "extracted-screenshots", "01_segment-1_error.png")
    ERR_SEG4 = os.path.join(ROOT, "docs", "extracted-screenshots", "04_segment-4_error.png")


configure_paths()

QW, QH = 540, 720
GW, GH = 1080, 1440
FPS = 24


def load_font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_SM = load_font(15, bold=True)
FONT_XS = load_font(12, bold=False)
FONT_LG = load_font(34, bold=True)
FONT_MD = load_font(20, bold=False)


def read_clip_frames_exact(path, start_frame, n_frames, target_w=QW, target_h=QH):
    """Read an exact contiguous frame range [start_frame, start_frame + n_frames) via trim=start_frame."""
    end_frame = start_frame + n_frames
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS,scale={target_w}:{target_h}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd)
    fb = target_w * target_h * 3
    frames = []
    for i in range(n_frames):
        chunk = raw[i * fb : (i + 1) * fb]
        if len(chunk) < fb:
            frames.append(frames[-1].copy() if frames else np.zeros((target_h, target_w, 3), dtype=np.uint8))
        else:
            frames.append(np.frombuffer(chunk, np.uint8).reshape(target_h, target_w, 3).copy())
    return frames


def read_clip_frames(path, start_sec, n_frames, target_w=QW, target_h=QH):
    return read_clip_frames_exact(path, int(round(start_sec * FPS)), n_frames, target_w, target_h)


def grab_frame_full(path, t_sec):
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{t_sec:.3f}", "-i", path,
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, np.uint8).reshape(960, 720, 3).copy()


def superellipse_poly(quad_pts, n_per_edge=12, radius_frac=0.14):
    """Project a rounded bezel rectangle into screen space via perspective homography."""
    w, h = 400.0, 260.0
    r = radius_frac * h
    pts = []
    corners = [
        (r, r, np.pi, 1.5 * np.pi),
        (w - r, r, 1.5 * np.pi, 2.0 * np.pi),
        (w - r, h - r, 0.0, 0.5 * np.pi),
        (r, h - r, 0.5 * np.pi, np.pi),
    ]
    for cx, cy, a0, a1 in corners:
        for a in np.linspace(a0, a1, n_per_edge):
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    src = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_quad = np.array(quad_pts, dtype=np.float32)
    src_quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    return np.round(warped).astype(np.int32)


def draw_tracker_overlay(frame_bgr, pose_idx, quad_list, scale_x=QW / 720.0, scale_y=QH / 960.0):
    out = frame_bgr.copy()
    q = quad_list[pose_idx] if 0 <= pose_idx < len(quad_list) else None
    if q is None:
        return out, "POSE %04d | OCCLUDED" % pose_idx

    q_np = np.array(q, dtype=np.float32)
    q_scaled = q_np.copy()
    q_scaled[:, 0] *= scale_x
    q_scaled[:, 1] *= scale_y

    bezel = superellipse_poly(q_scaled)
    cv2.polylines(out, [bezel], True, (74, 162, 201), 2, lineType=cv2.LINE_AA)

    poly = np.round(q_scaled).astype(np.int32)
    cv2.polylines(out, [poly], True, (143, 185, 95), 2, lineType=cv2.LINE_AA)
    for pt in poly:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < QW and 0 <= py < QH:
            cv2.circle(out, (px, py), 4, (143, 185, 95), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (px, py), 5, (14, 13, 11), 1, lineType=cv2.LINE_AA)

    tl = q_np[0]
    tr = q_np[1]
    clip_tag = " [CLIP->PROJ]" if tr[0] > 720.0 else ""
    coord_txt = f"P{pose_idx:04d} TL({tl[0]:.0f},{tl[1]:.0f}){clip_tag}"
    return out, coord_txt


def draw_hud_badge(img_bgr, title, subtext="", border_rgb=(95, 185, 143)):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil, "RGBA")
    pad_x, pad_y = 12, 8
    x0, y0 = 14, 14
    tb1 = dr.textbbox((0, 0), title, font=FONT_SM)
    tw = tb1[2] - tb1[0]
    th = tb1[3] - tb1[1]
    if subtext:
        tb2 = dr.textbbox((0, 0), subtext, font=FONT_XS)
        tw = max(tw, tb2[2] - tb2[0])
        th += (tb2[3] - tb2[1]) + 6
    dr.rounded_rectangle(
        [x0, y0, x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y],
        radius=8,
        fill=(14, 13, 11, 210),
        outline=border_rgb + (210,),
        width=1,
    )
    dr.text((x0 + pad_x, y0 + pad_y - 1), title, font=FONT_SM, fill=(242, 239, 233, 255))
    if subtext:
        dr.text((x0 + pad_x, y0 + pad_y + 21), subtext, font=FONT_XS, fill=border_rgb + (255,))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def build_debug_video():
    with open(QUADS_PATH) as f:
        quads = json.load(f)["quads"]

    n_seg = 120
    windows = [
        {
            "label": "WIN 1 | SEG 1 BEZEL LOCK",
            "tl_src": (STORY288, 10.0),
            "tr_src": (DEFECTIVE, 10.0),
            "bl_src": (STORY288, 10.0),
            "br_src": (NANSEN, 10.5),
            "br_tag": "EP 1: NANSEN SILVER-ICE",
        },
        {
            "label": "WIN 2 | SEG 4 CLIPPED PAN",
            "tl_src": (STORY288, 202.0),
            "tr_src": (DEFECTIVE, 202.0),
            "bl_src": (STORY288, 202.0),
            "br_src": (SHACKLETON, 135.8),
            "br_tag": "EP 2: SHACKLETON AMBER NOIR",
        },
        {
            "label": "WIN 3 | MIRROR TO CUBIST AIRLOCK",
            "tl_src": (STORY288, 265.0),
            "tr_src": (DEFECTIVE, 265.0),
            "bl_src": (STORY288, 239.0),
            "br_src": (MARS500, 233.5),
            "br_tag": "EP 3: MARS500 CUBIST AIRLOCK",
        },
    ]

    temp_silent = os.path.join(ROOT, "work", "_tmp_debug_2x2_silent.mp4")
    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{GW}x{GH}", "-r", str(FPS), "-i", "-",
            "-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-r", str(FPS), temp_silent,
        ],
        stdin=subprocess.PIPE,
    )

    for w in windows:
        tl_f = read_clip_frames(w["tl_src"][0], w["tl_src"][1], n_seg)
        tr_f = read_clip_frames(w["tr_src"][0], w["tr_src"][1], n_seg)
        br_f = read_clip_frames(w["br_src"][0], w["br_src"][1], n_seg)

        bl_start_f = int(round(w["bl_src"][1] * FPS))
        bl_even0 = bl_start_f - (bl_start_f % 2)
        bl_f_raw = read_clip_frames_exact(w["bl_src"][0], bl_even0, n_seg + 2)
        for i in range(n_seg):
            cur_f = bl_start_f + i
            even_f = cur_f - (cur_f % 2)
            local_even_idx = max(0, min(len(bl_f_raw) - 1, even_f - bl_even0))
            pose_idx = min(len(quads) - 1, even_f // 2)

            p_tl = draw_hud_badge(tl_f[i], "01 / RAW POV CAPTURE", w["label"], (154, 149, 140))
            p_tr = draw_hud_badge(tr_f[i], "02 / FAILED 3-DoF MATCH", "UNCONSTRAINED TEMPLATE DRIFT", (224, 108, 117))
            bl_ovr, coord_txt = draw_tracker_overlay(bl_f_raw[local_even_idx], pose_idx, quads)
            p_bl = draw_hud_badge(bl_ovr, "03 / 8-DoF HUBER + LK TRACK", coord_txt, (95, 185, 143))
            p_br = draw_hud_badge(br_f[i], "04 / NIGHT-CHALK TRILOGY", w["br_tag"], (201, 162, 74))

            canvas = np.full((GH, GW, 3), 14, dtype=np.uint8)
            canvas[0:QH, 0:QW] = p_tl
            canvas[0:QH, QW:GW] = p_tr
            canvas[QH:GH, 0:QW] = p_bl
            canvas[QH:GH, QW:GW] = p_br
            cv2.line(canvas, (QW, 0), (QW, GH), (14, 13, 11), 4)
            cv2.line(canvas, (0, QH), (GW, QH), (14, 13, 11), 4)
            enc.stdin.write(canvas.tobytes())

    enc.stdin.close()
    assert enc.wait() == 0, "debug video encode failed"

    os.makedirs(os.path.dirname(OUT_LINKEDIN), exist_ok=True)
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error",
        "-i", temp_silent,
        "-ss", "34.0", "-t", "15.0", "-i", AUDIO_WAV,
        "-af", "afade=t=in:st=0:d=0.8,afade=t=out:st=13.5:d=1.5",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-shortest", OUT_LINKEDIN,
    ])
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error",
        "-i", OUT_LINKEDIN,
        "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-maxrate", "2800k", "-bufsize", "5600k",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", OUT_WEB,
    ])
    os.remove(temp_silent)
    print(f"Wrote {OUT_LINKEDIN} ({os.path.getsize(OUT_LINKEDIN)} B)")
    print(f"Wrote {OUT_WEB} ({os.path.getsize(OUT_WEB)} B)")


def _crop_dash_focus(bgr_720x960, target_w=680, target_h=360):
    h, w = bgr_720x960.shape[:2]
    if (w, h) != (720, 960):
        bgr_720x960 = cv2.resize(bgr_720x960, (720, 960), interpolation=cv2.INTER_AREA)
    crop = bgr_720x960[460:840, 80:720]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _crop_screenshot_dash(path, target_w=680, target_h=360):
    im = cv2.imread(path)
    assert im is not None, f"Missing screenshot: {path}"
    h, w = im.shape[:2]
    y0 = int(round(h * 0.44))
    y1 = int(round(h * 0.88))
    x0 = int(round(w * 0.14))
    crop = im[y0:y1, x0:w]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)


def build_tracker_comparison():
    seg1_bad = _crop_screenshot_dash(ERR_SEG1)
    seg4_bad = _crop_screenshot_dash(ERR_SEG4)
    seg1_fix = _crop_dash_focus(grab_frame_full(NANSEN, 12.15))
    seg4_fix = _crop_dash_focus(grab_frame_full(SHACKLETON, 138.15))

    pad = 16
    tw, th = 680, 360
    cw = tw * 2 + pad * 3
    ch = th * 2 + pad * 3
    board = Image.new("RGB", (cw, ch), (14, 13, 11))

    items = [
        (seg1_bad, (pad, pad), "BEFORE: SEG 1 (0:12) | 3-DoF RECTANGLE SPILLS UI", (224, 108, 117)),
        (seg1_fix, (pad * 2 + tw, pad), "AFTER: SEG 1 | 8-DoF HUBER + SUPERELLIPSE MASK", (95, 185, 143)),
        (seg4_bad, (pad, pad * 2 + th), "BEFORE: SEG 4 (3:24) | SKEW OVERFLOWS BEZEL", (224, 108, 117)),
        (seg4_fix, (pad * 2 + tw, pad * 2 + th), "AFTER: SEG 4 | RIGHT-CLIPPED HEAD PAN LOCK", (95, 185, 143)),
    ]
    for bgr, (x, y), lbl, col in items:
        badged = draw_hud_badge(bgr, lbl, border_rgb=col)
        rgb = Image.fromarray(cv2.cvtColor(badged, cv2.COLOR_BGR2RGB))
        board.paste(rgb, (x, y))

    out_p = os.path.join(SITE_NOTE_DIR, "tracker-comparison.jpg")
    board.save(out_p, quality=92)
    print(f"Wrote {out_p}")


def _resolve_episode_thumbnails():
    """Return the three 16:9 YouTube episode thumbnail paths in episode order."""
    specs = [
        ("thumb-ep01-nansen.jpg", "yt_16x9_01_*.jpg"),
        ("thumb-ep02-shackleton.jpg", "yt_16x9_02_*.jpg"),
        ("thumb-ep03-mars500.jpg", "yt_16x9_03_*.jpg"),
    ]
    repo_images = paths.REPO_ROOT / "docs" / "images"
    out = []
    for doc_name, raw_glob in specs:
        candidate = repo_images / doc_name
        if candidate.is_file():
            out.append(str(candidate))
            continue
        matches = sorted((Path(ROOT) / "output" / "thumbnails").glob(raw_glob))
        if not matches:
            raise FileNotFoundError(f"Missing episode thumbnail: {candidate} or {raw_glob}")
        out.append(str(matches[0]))
    return out


def build_trilogy_palettes():
    """Compose the three prepared 16:9 YouTube thumbnails side-by-side."""
    pad = 16
    tw, th = 640, 360
    cw = pad + (tw + pad) * 3
    ch = th + pad * 2
    strip = Image.new("RGB", (cw, ch), (14, 13, 11))
    dr = ImageDraw.Draw(strip)

    for idx, thumb_path in enumerate(_resolve_episode_thumbnails()):
        with Image.open(thumb_path) as im:
            card = im.convert("RGB").resize((tw, th), Image.LANCZOS)
        x = pad + idx * (tw + pad)
        y = pad
        strip.paste(card, (x, y))
        dr.rectangle([x - 1, y - 1, x + tw, y + th], outline=(55, 52, 46), width=1)

    out_p = os.path.join(SITE_NOTE_DIR, "trilogy-palettes.jpg")
    strip.save(out_p, quality=93)
    print(f"Wrote {out_p} ({cw}x{ch})")


def build_og_image():
    W, H = 1200, 627
    img = Image.new("RGB", (W, H), (14, 13, 11))
    dr = ImageDraw.Draw(img)

    for y in range(28, H, 34):
        for x in range(28, 760, 34):
            dr.ellipse([x, y, x + 2, y + 2], fill=(35, 33, 29))

    dr.ellipse([64, 72, 74, 82], fill=(95, 185, 143))
    dr.text((86, 68), "CARLOS CRESPO MACAYA  |  NOTES", font=FONT_SM, fill=(95, 185, 143))
    title_lines = [
        "Night-Chalk Drive: a 12-fps",
        "rotoscope pipeline from",
        "Meta Ray-Ban night footage",
    ]
    ty = 125
    for line in title_lines:
        dr.text((64, ty), line, font=FONT_LG, fill=(242, 239, 233))
        ty += 48

    dr.text((64, 305), "OpenCV 8-DoF tracking, CIE-Lab night", font=FONT_MD, fill=(154, 149, 140))
    dr.text((64, 335), "shading, and 6-DoF 3D portals.", font=FONT_MD, fill=(154, 149, 140))

    metrics = [
        ("6,720", "FINAL FRAMES"),
        ("874", "LOCKED LCD POSES"),
        ("0.00 px", "WITHIN-PAIR JITTER"),
    ]
    mx = 64
    for val, lbl in metrics:
        dr.rounded_rectangle([mx, 445, mx + 185, 535], radius=10, fill=(22, 21, 18), outline=(55, 52, 46), width=1)
        dr.text((mx + 16, 462), val, font=load_font(24, bold=True), fill=(201, 162, 74))
        dr.text((mx + 16, 500), lbl, font=FONT_XS, fill=(154, 149, 140))
        mx += 200

    px, pw, ph, gap = 808, 328, 184, 12
    for idx, thumb_path in enumerate(_resolve_episode_thumbnails()):
        with Image.open(thumb_path) as im:
            card = im.convert("RGB").resize((pw, ph), Image.LANCZOS)
        y_pos = 25 + idx * (ph + gap)
        img.paste(card, (px, y_pos))
        dr.rectangle([px - 1, y_pos - 1, px + pw, y_pos + ph], outline=(55, 52, 46), width=1)

    out_p = os.path.join(SITE_NOTE_DIR, "og.png")
    img.save(out_p)
    print(f"Wrote {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Build note article media and OpenGraph preview assets.")
    parser.add_argument("--data-dir", default=None, help="Root folder containing work/ and output/ media")
    parser.add_argument("--site-note-dir", default=None, help="Destination directory for website note assets")
    parser.add_argument("--only-og", action="store_true", help="Only regenerate og.png")
    parser.add_argument(
        "--only-trilogy-and-og",
        action="store_true",
        help="Regenerate trilogy-palettes.jpg and og.png from episode thumbnails",
    )
    args = parser.parse_args()

    configure_paths(data_dir=args.data_dir, site_note_dir=args.site_note_dir)
    os.makedirs(SITE_NOTE_DIR, exist_ok=True)
    if args.only_og:
        build_og_image()
        return
    if args.only_trilogy_and_og:
        build_trilogy_palettes()
        build_og_image()
        return

    build_debug_video()
    build_tracker_comparison()
    build_trilogy_palettes()
    build_og_image()


if __name__ == "__main__":
    main()
