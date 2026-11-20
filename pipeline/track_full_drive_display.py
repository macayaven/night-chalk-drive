#!/usr/bin/env python3
"""Track the car's center infotainment display across ALL 3,198 12-fps poses
of work/final3_parts/head_raw.mp4 (both full 4-corner and right-clipped views).

Seeds exact verified quads for Panels 1..6 from data/track_quads_expedition.json,
detects all other visible screen poses across 0..3197, propagates/bridges short
motion-blur gaps via forward-backward Lucas-Kanade homographies, pads outward to
cover the LCD glow halo, smooths within contiguous visibility segments, and writes:
  - data/track_quads_full_drive.json
  - work/track_diag/full_drive_quads_verify.jpg

Stage persistence rule (`stage_for_pose`):
  Stage 1: pose_idx < 1094            (0.00s  .. 91.15s, through P1 until P2)
  Stage 2: 1094 <= pose_idx < 1697    (91.15s .. 141.40s, from P2 until P3)
  Stage 3: 1697 <= pose_idx < 2450    (141.40s .. 204.20s, from P3 until P4)
  Stage 4: 2450 <= pose_idx < 2850    (204.20s .. 237.50s, from P4 until P5)
  Stage 5: 2850 <= pose_idx < 2888    (237.50s .. 240.65s, from P5 until P6)
  Stage 6: pose_idx >= 2888           (240.65s .. end of drive)
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import media_base as B, night_chalk_shader as R, paths, planar_tracker_segments as TP

W, H = R.W, R.H
N_SRC_POSES = 3198
EXP_QUADS_PATH = paths.QUADS_EXPEDITION
OUT_JSON_PATH = paths.QUADS_FULL_DRIVE
VERIFY_IMG_PATH = os.path.join(B.WORK, "track_diag", "full_drive_quads_verify.jpg")

HEAD_SIGNS = [
    (1, 11.90, 12.60),
    (2, 91.15, 91.85),
    (3, 141.40, 141.85),
    (4, 204.20, 204.60),
    (5, 237.50, 238.15),
    (6, 240.65, 241.35),
]

# Start poses for P2..P6: int(round(st * 24)) // 2
STAGE_BOUNDS = [
    (1, 0, 1094),
    (2, 1094, 1697),
    (3, 1697, 2450),
    (4, 2450, 2850),
    (5, 2850, 2888),
    (6, 2888, N_SRC_POSES),
]

# Empirical perspective median across verified P1..P6 quads
MED_AR = 1.535
MED_UT = np.array([0.9965, -0.055], dtype=np.float32)
MED_UT /= np.linalg.norm(MED_UT)
MED_UB = np.array([0.9945, -0.092], dtype=np.float32)
MED_UB /= np.linalg.norm(MED_UB)

# Roi bounds in 720x960 frame containing the center infotainment display
ROI_Y0, ROI_Y1 = 490, 850
ROI_X0, ROI_X1 = 295, 720

K7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
K5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
K3 = np.ones((3, 3), np.uint8)
K_DIL_OUT = np.ones((21, 21), np.uint8)
K_DIL_IN = np.ones((9, 9), np.uint8)


def stage_for_pose(pose_idx: int) -> int:
    for stg, s0, s1 in STAGE_BOUNDS:
        if s0 <= pose_idx < s1:
            return stg
    return 6


def load_anchor_poses():
    """Return dict pose_idx -> (panel, quad_4x2_f32) from verified track_quads_expedition.json."""
    with open(EXP_QUADS_PATH) as f:
        tq = json.load(f)
    anchors = {}
    for p, st, _ in HEAD_SIGNS:
        sf = int(round(st * 24.0))
        quads = tq[str(p)]["quads"]
        for i, q in enumerate(quads):
            p_idx = (sf + i) // 2
            if q is not None and p_idx not in anchors:
                anchors[p_idx] = (p, np.array(q, dtype=np.float32))
    return anchors


def _fit_screen_quad_from_contour(cnt, bx, by, bw, bh, val_roi):
    """Fit [tl, tr, br, bl] in full frame coords from a contour in ROI coords."""
    hull = cv2.convexHull(cnt)
    m = np.zeros(val_roi.shape, np.uint8)
    cv2.drawContours(m, [hull], -1, 255, -1)

    right_clipped = (bx + bw >= ROI_X1 - 6)  # touches right edge (x >= 714)

    # Fit left edge in full coordinates
    y0 = max(by + 3, int(round(by + 0.16 * bh)))
    y1 = min(by + bh - 3, int(round(by + 0.84 * bh)))
    left_pts, right_pts = [], []
    for y in range(y0, y1 + 1):
        ry = y - ROI_Y0
        if not (0 <= ry < m.shape[0]):
            continue
        cols = np.where(m[ry, :] > 0)[0]
        if len(cols) >= 12:
            left_pts.append([float(cols[0] + ROI_X0), float(y)])
            right_pts.append([float(cols[-1] + ROI_X0), float(y)])
    if len(left_pts) < 6:
        return None
    ll = TP.fit_line_2d(left_pts)
    if ll[1] < 0:
        ll = -ll * np.array([1.0, 1.0, -1.0, -1.0])
    # Ensure direction vector points downward
    if ll[1] < 0:
        ll[:2] = -ll[:2]

    if not right_clipped:
        if len(right_pts) < 6:
            return None
        lr = TP.fit_line_2d(right_pts)
        if lr[1] < 0:
            lr[:2] = -lr[:2]

        x0 = max(bx + 3, int(round(bx + 0.16 * bw)))
        x1 = min(bx + bw - 3, int(round(bx + 0.84 * bw)))
        top_pts, bot_pts = [], []
        for x in range(x0, x1 + 1):
            rx = x - ROI_X0
            if not (0 <= rx < m.shape[1]):
                continue
            rows = np.where(m[:, rx] > 0)[0]
            if len(rows) >= 20:
                top_pts.append([float(x), float(rows[0] + ROI_Y0)])
                bot_pts.append([float(x), float(rows[-1] + ROI_Y0)])
        if len(top_pts) < 8 or len(bot_pts) < 8:
            return None
        lt = TP.fit_line_2d(top_pts)
        lb = TP.fit_line_2d(bot_pts)
        if lt[0] < 0:
            lt[:2] = -lt[:2]
        if lb[0] < 0:
            lb[:2] = -lb[:2]

        tl = TP.intersect_lines(ll, lt)
        tr = TP.intersect_lines(lt, lr)
        br = TP.intersect_lines(lr, lb)
        bl = TP.intersect_lines(lb, ll)
        if any(p is None for p in (tl, tr, br, bl)):
            return None

        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_l = float(np.linalg.norm(bl - tl))
        h_r = float(np.linalg.norm(br - tr))
        if not (118.0 <= w_top <= 174.0 and 118.0 <= w_bot <= 174.0):
            return None
        if not (72.0 <= h_l <= 124.0 and 72.0 <= h_r <= 124.0):
            return None
        ar = 0.5 * (w_top + w_bot) / (0.5 * (h_l + h_r) + 1e-6)
        if not (1.24 <= ar <= 1.74):
            return None
        # Reject skewed trapezoids / non-rectangles
        u_t = (tr - tl) / (w_top + 1e-6)
        u_l = (bl - tl) / (h_l + 1e-6)
        if abs(float(np.dot(u_t, u_l))) > 0.28:
            return None
        return np.array([tl, tr, br, bl], dtype=np.float32), False

    # Right-clipped screen: visible left/top/bottom edges; project right edge
    if bx < 450 or bx > 702 or bw < 18 or not (68.0 <= bh <= 132.0):
        return None
    x0 = max(bx + 2, int(round(bx + 0.14 * bw)))
    x1 = min(bx + bw - 3, int(round(bx + 0.86 * bw)))
    top_pts, bot_pts = [], []
    for x in range(x0, x1 + 1):
        rx = x - ROI_X0
        if not (0 <= rx < m.shape[1]):
            continue
        rows = np.where(m[:, rx] > 0)[0]
        if len(rows) >= 20:
            top_pts.append([float(x), float(rows[0] + ROI_Y0)])
            bot_pts.append([float(x), float(rows[-1] + ROI_Y0)])

    if len(top_pts) >= 14 and len(bot_pts) >= 14 and bw >= 42:
        lt = TP.fit_line_2d(top_pts)
        lb = TP.fit_line_2d(bot_pts)
        if lt[0] < 0:
            lt[:2] = -lt[:2]
        if lb[0] < 0:
            lb[:2] = -lb[:2]
        u_top = np.array([lt[0], lt[1]], dtype=np.float32)
        u_bot = np.array([lb[0], lb[1]], dtype=np.float32)
        # Guard against steep slope distortion from partial hand/dash glint
        if abs(u_top[1]) > 0.32 or u_top[0] < 0.90:
            u_top = MED_UT.copy()
        if abs(u_bot[1]) > 0.34 or u_bot[0] < 0.90:
            u_bot = MED_UB.copy()
        lt_use = np.array([u_top[0], u_top[1], lt[2], lt[3]], dtype=np.float64)
        lb_use = np.array([u_bot[0], u_bot[1], lb[2], lb[3]], dtype=np.float64)
        tl = TP.intersect_lines(ll, lt_use)
        bl = TP.intersect_lines(ll, lb_use)
    else:
        # Very narrow right slice (< 42px wide): anchor TL/BL on median top/bot near left edge
        if len(top_pts) < 3 or len(bot_pts) < 3:
            return None
        y_t = float(np.median([p[1] for p in top_pts]))
        y_b = float(np.median([p[1] for p in bot_pts]))
        u_top = MED_UT.copy()
        u_bot = MED_UB.copy()
        # Intersect horizontal y=y_t / y=y_b with left line ll
        tl = TP.intersect_lines(ll, np.array([u_top[0], u_top[1], np.median([p[0] for p in top_pts]), y_t], dtype=np.float64))
        bl = TP.intersect_lines(ll, np.array([u_bot[0], u_bot[1], np.median([p[0] for p in bot_pts]), y_b], dtype=np.float64))

    if tl is None or bl is None:
        return None
    h_left = float(np.linalg.norm(bl - tl))
    if not (70.0 <= h_left <= 124.0):
        return None
    w_proj = MED_AR * h_left
    tr = tl + u_top * w_proj
    br = bl + u_bot * w_proj
    return np.array([tl, tr, br, bl], dtype=np.float32), True


def _score_candidate(q, is_clipped, val_roi, fill_area, bbox_area):
    """Score candidate quad by oriented polygon fill and bezel contrast; return None if invalid."""
    bbox_fill = fill_area / (bbox_area + 1e-6)
    if bbox_fill < 0.62:
        return None

    m = np.zeros((ROI_Y1 - ROI_Y0, ROI_X1 - ROI_X0), np.uint8)
    q_roi = np.round(q - np.array([[ROI_X0, ROI_Y0]], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(m, [q_roi], 255)
    vis_quad_area = float((m > 0).sum())
    if vis_quad_area < 800.0:
        return None
    poly_fill = min(1.0, fill_area / (vis_quad_area + 1e-6))
    if poly_fill < 0.72:
        return None

    m_in = cv2.erode(m, K_DIL_IN) > 0
    if m_in.sum() < 120:
        m_in = m > 0
    if m_in.sum() < 40:
        return None

    # Dark bezel ring on left, top, bottom outside LCD
    ring = (cv2.dilate(m, K_DIL_OUT) > 0) & ~(cv2.dilate(m, K7) > 0)
    if ring.sum() < 80:
        return None

    in_med = float(np.median(val_roi[m_in]))
    in_p15 = float(np.percentile(val_roi[m_in], 15))
    out_p35 = float(np.percentile(val_roi[ring], 35))
    contrast = in_med - out_p35

    # Must be bright LCD enclosed by dark matte dashboard bezel (reject dark off-screen or garage walls)
    if in_med < 100.0 or in_p15 < 42.0 or out_p35 > 58.0 or contrast < 52.0:
        return None

    w_q = 0.5 * (float(np.linalg.norm(q[1] - q[0])) + float(np.linalg.norm(q[2] - q[3])))
    h_q = 0.5 * (float(np.linalg.norm(q[3] - q[0])) + float(np.linalg.norm(q[2] - q[1])))
    ar = w_q / (h_q + 1e-6)
    ar_bonus = 1.0 - min(1.0, abs(ar - MED_AR) * 2.0)
    clip_pen = 0.88 if is_clipped else 1.0
    score = contrast * poly_fill * (0.7 + 0.3 * ar_bonus) * clip_pen
    return float(score)


def detect_screen_pose(fr):
    """Detect the infotainment LCD quad [tl, tr, br, bl] on a single raw frame, or None."""
    roi = fr[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
    val_raw = roi.max(axis=2)

    # Secondary skin-suppressed luminance for cases where driver hand/wrist touches screen edge
    b = roi[:, :, 0].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)
    warm_skin = (r - b > 34) & (b < 155) & (val_raw < 185)
    val_clean = val_raw.copy()
    val_clean[warm_skin] = np.minimum(val_clean[warm_skin], 35)

    best_q = None
    best_score = -1.0

    for val_map in (val_raw, val_clean):
        closed7 = cv2.morphologyEx(val_map, cv2.MORPH_CLOSE, K7)
        closed5 = cv2.morphologyEx(val_map, cv2.MORPH_CLOSE, K5)
        for src_m, open_k, vthr in [
            (closed5, K5, 88),
            (closed7, K3, 84),
            (closed7, K3, 72),
            (closed7, K3, 60),
        ]:
            th = cv2.morphologyEx((src_m > vthr).astype(np.uint8) * 255, cv2.MORPH_OPEN, open_k)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area < 1200 or area > 21000:
                    continue
                bx_r, by_r, bw, bh = cv2.boundingRect(c)
                bx, by = bx_r + ROI_X0, by_r + ROI_Y0
                if by < 495 or by > 735 or by + bh > 840:
                    continue
                if bh < 66 or bh > 132 or bw > 180:
                    continue
                if area / (bw * bh + 1e-6) < 0.62:
                    continue

                fit = _fit_screen_quad_from_contour(c, bx, by, bw, bh, val_raw)
                if fit is None:
                    continue
                q, is_clipped = fit
                sc = _score_candidate(q, is_clipped, val_raw, area, bw * bh)
                if sc is not None and sc > best_score:
                    best_score = sc
                    best_q = q
        # Avoid re-running skin-suppressed map if standard luminance already gave a clean match
        if best_q is not None:
            break

    return best_q


def verify_quad_visible(fr, q):
    """Confirm an LK-propagated quad still lands on an illuminated screen within frame."""
    if q is None:
        return False
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 255)
    # Must have at least 16 px visible width in [0..720)
    vis_xs = np.where(m.any(axis=0))[0]
    if len(vis_xs) < 16:
        return False
    m_in = cv2.erode(m, K_DIL_IN) > 0
    if m_in.sum() < 80:
        m_in = m > 0
    if m_in.sum() < 40:
        return False
    ring = (cv2.dilate(m, K_DIL_OUT) > 0) & ~(cv2.dilate(m, K7) > 0)
    ring[:ROI_Y0, :] = False
    if ring.sum() < 40:
        return False
    val = fr.max(axis=2)
    in_med = float(np.median(val[m_in]))
    out_p35 = float(np.percentile(val[ring], 35))
    return (in_med >= 90.0) and (out_p35 <= 60.0) and ((in_med - out_p35) >= 42.0)


def main():
    os.makedirs(os.path.dirname(VERIFY_IMG_PATH), exist_ok=True)
    anchors = load_anchor_poses()
    print(f"Loaded {len(anchors)} exact P1..P6 anchor poses.")

    # Single sequential pass reading all 3,198 12-fps poses (keep frames in memory ~ 3198 * 2MB ~ 6.6GB? No: stream pass1 -> detect, only keep small neighbor window or 2nd pass for LK)
    raw_det = [None] * N_SRC_POSES
    locked = [False] * N_SRC_POSES

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", R.HEAD_RAW, "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = W * H * 3
    n_det = 0
    for src_f in range(N_SRC_POSES * 2):
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        if src_f % 2 != 0:
            continue
        p_idx = src_f // 2
        if p_idx in anchors:
            raw_det[p_idx] = anchors[p_idx][1].copy()
            locked[p_idx] = True
            n_det += 1
            continue
        fr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
        q = detect_screen_pose(fr)
        if q is not None:
            raw_det[p_idx] = q
            n_det += 1
        if (p_idx + 1) % 500 == 0:
            print(f"Pass 1 detect: {p_idx + 1}/{N_SRC_POSES} (matched={n_det})", flush=True)
    dec.stdout.close()
    dec.wait()
    print(f"Pass 1 raw detections + anchors: {n_det} / {N_SRC_POSES}")

    # Prune isolated 1-pose false positives (keep >=2-pose runs and locked anchors)
    pruned_single = 0
    for i in range(N_SRC_POSES):
        if locked[i] or raw_det[i] is None:
            continue
        has_prev = i > 0 and raw_det[i - 1] is not None
        has_next = i + 1 < N_SRC_POSES and raw_det[i + 1] is not None
        if not has_prev and not has_next:
            raw_det[i] = None
            pruned_single += 1
    print(f"Pruned {pruned_single} isolated 1-pose glitches.")

    # Identify LK propagation targets: short interior gaps (1..6 poses) + 1..2 boundary poses adjacent to detections
    lk_need_poses = set()
    i = 0
    while i < N_SRC_POSES:
        if raw_det[i] is not None:
            i += 1
            continue
        j = i
        while j < N_SRC_POSES and raw_det[j] is None:
            j += 1
        gap_len = j - i
        left_ok = i > 0 and raw_det[i - 1] is not None
        right_ok = j < N_SRC_POSES and raw_det[j] is not None
        if left_ok and right_ok and gap_len <= 6:
            for k in range(i - 1, j + 1):
                lk_need_poses.add(k)
        else:
            # Check 2 poses before a run starts / 2 poses after a run ends when near visible edge
            if right_ok and float(raw_det[j][0, 0]) < 706.0:
                for k in range(max(0, j - 2), j + 1):
                    lk_need_poses.add(k)
            if left_ok and float(raw_det[i - 1][0, 0]) < 706.0:
                for k in range(i - 1, min(N_SRC_POSES, i + 2)):
                    lk_need_poses.add(k)
        i = j

    # Second quick sequential pass caching ONLY frames needed for LK gap/boundary propagation
    cached_frames = {}
    if lk_need_poses:
        max_need = max(lk_need_poses)
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", R.HEAD_RAW, "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE,
        )
        for src_f in range((max_need + 1) * 2):
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if src_f % 2 != 0:
                continue
            p_idx = src_f // 2
            if p_idx in lk_need_poses:
                cached_frames[p_idx] = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
        dec.stdout.close()
        dec.wait()

    # Fill verified short interior gaps + verified edge frames
    recovered_gaps = 0
    recovered_edges = 0
    # Copy raw_det to quads_out so we don't chain unverified frames
    quads_out = [q.copy() if q is not None else None for q in raw_det]

    i = 0
    while i < N_SRC_POSES:
        if raw_det[i] is not None:
            i += 1
            continue
        j = i
        while j < N_SRC_POSES and raw_det[j] is None:
            j += 1
        gap_len = j - i
        left_ok = i > 0 and raw_det[i - 1] is not None
        right_ok = j < N_SRC_POSES and raw_det[j] is not None
        if left_ok and right_ok and gap_len <= 6:
            # Propagate forward from i-1 and backward from j
            fwd_q = {}
            q_cur = raw_det[i - 1].copy()
            for k in range(i, j):
                if (k - 1) in cached_frames and k in cached_frames:
                    Hk = TP.compute_lk_homography(cached_frames[k - 1], cached_frames[k], q_cur)
                    if Hk is not None:
                        q_cur = TP.warp_quad(q_cur, Hk)
                        fwd_q[k] = q_cur.copy()
                    else:
                        break
            bwd_q = {}
            q_cur = raw_det[j].copy()
            for k in range(j - 1, i - 1, -1):
                if (k + 1) in cached_frames and k in cached_frames:
                    Hk = TP.compute_lk_homography(cached_frames[k + 1], cached_frames[k], q_cur)
                    if Hk is not None:
                        q_cur = TP.warp_quad(q_cur, Hk)
                        bwd_q[k] = q_cur.copy()
                    else:
                        break
            for k in range(i, j):
                cand = None
                if k in fwd_q and k in bwd_q:
                    w_b = (k - (i - 1)) / float(gap_len + 1)
                    cand = (1.0 - w_b) * fwd_q[k] + w_b * bwd_q[k]
                elif k in fwd_q:
                    cand = fwd_q[k]
                elif k in bwd_q:
                    cand = bwd_q[k]
                elif gap_len <= 2:
                    w_b = (k - (i - 1)) / float(gap_len + 1)
                    cand = (1.0 - w_b) * raw_det[i - 1] + w_b * raw_det[j]
                if cand is not None and k in cached_frames and verify_quad_visible(cached_frames[k], cand):
                    quads_out[k] = cand.astype(np.float32)
                    recovered_gaps += 1
        else:
            if right_ok:
                q_cur = raw_det[j].copy()
                for k in (j - 1, j - 2):
                    if k >= 0 and raw_det[k] is None and (k + 1) in cached_frames and k in cached_frames:
                        Hk = TP.compute_lk_homography(cached_frames[k + 1], cached_frames[k], q_cur)
                        if Hk is None:
                            break
                        q_cur = TP.warp_quad(q_cur, Hk)
                        if verify_quad_visible(cached_frames[k], q_cur):
                            quads_out[k] = q_cur.astype(np.float32)
                            recovered_edges += 1
                        else:
                            break
            if left_ok:
                q_cur = raw_det[i - 1].copy()
                for k in (i, i + 1):
                    if k < N_SRC_POSES and raw_det[k] is None and (k - 1) in cached_frames and k in cached_frames:
                        Hk = TP.compute_lk_homography(cached_frames[k - 1], cached_frames[k], q_cur)
                        if Hk is None:
                            break
                        q_cur = TP.warp_quad(q_cur, Hk)
                        if verify_quad_visible(cached_frames[k], q_cur):
                            quads_out[k] = q_cur.astype(np.float32)
                            recovered_edges += 1
                        else:
                            break
        i = j

    print(f"LK recovered: {recovered_gaps} gap poses, {recovered_edges} edge poses.")

    # Pad outward on non-anchor poses to match track_display_planar bezel coverage
    padded = []
    for i in range(N_SRC_POSES):
        q = quads_out[i]
        if q is None:
            padded.append(None)
        elif locked[i]:
            padded.append(q.copy())
        else:
            padded.append(TP.pad_quad_outward(q, 2.2, 2.5))

    # Gentle zero-phase temporal smoothing inside contiguous runs (anchors stay exact)
    smoothed = [q.copy() if q is not None else None for q in padded]
    for i in range(1, N_SRC_POSES - 1):
        if locked[i] or padded[i] is None:
            continue
        if padded[i - 1] is not None and padded[i + 1] is not None:
            d_prev = float(np.max(np.abs(padded[i] - padded[i - 1])))
            d_next = float(np.max(np.abs(padded[i + 1] - padded[i])))
            if d_prev < 35.0 and d_next < 35.0:
                smoothed[i] = (0.20 * padded[i - 1] + 0.60 * padded[i] + 0.20 * padded[i + 1]).astype(np.float32)

    final_count = sum(1 for q in smoothed if q is not None)
    # Count contiguous runs
    runs = []
    cur_s = None
    for i in range(N_SRC_POSES):
        if smoothed[i] is not None and cur_s is None:
            cur_s = i
        elif smoothed[i] is None and cur_s is not None:
            runs.append((cur_s, i - 1))
            cur_s = None
    if cur_s is not None:
        runs.append((cur_s, N_SRC_POSES - 1))

    print(f"Final tracked full-drive poses: {final_count} / {N_SRC_POSES} across {len(runs)} contiguous runs.")
    for stg, s0, s1 in STAGE_BOUNDS:
        cnt_s = sum(1 for p in range(s0, s1) if smoothed[p] is not None)
        print(f"  Stage {stg} (poses {s0:4d}..{s1 - 1:4d}): {cnt_s:4d} visible screen poses")

    # Save JSON
    payload = {
        "n_poses": N_SRC_POSES,
        "stage_bounds": [list(x) for x in STAGE_BOUNDS],
        "final_visible_poses": final_count,
        "quads": [q.tolist() if q is not None else None for q in smoothed],
        "stages": [stage_for_pose(i) if smoothed[i] is not None else 0 for i in range(N_SRC_POSES)],
    }
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {OUT_JSON_PATH}")

    # Render 36-tile visual verification sheet across all 6 stages (mix of full, right-clipped, and early/late frames)
    sample_candidates = [
        4, 15, 42, 68, 118, 137, 145, 185, 226, 244, 379, 574,
        796, 875, 900, 1085, 1100, 1154, 1226, 1303, 1495, 1540, 1659, 1701,
        1760, 1804, 1983, 2188, 2328, 2367, 2454, 2465, 2852, 2888, 2912, 2976,
    ]
    sample_set = set(sample_candidates)
    tiles = []
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", R.HEAD_RAW, "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    for src_f in range(N_SRC_POSES * 2):
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        if src_f % 2 != 0:
            continue
        p_idx = src_f // 2
        if p_idx in sample_set:
            fr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
            q = smoothed[p_idx]
            stg = stage_for_pose(p_idx)
            if q is not None:
                cv2.polylines(fr, [np.round(q).astype(np.int32)], True, (0, 255, 0), 3)
                col = (0, 255, 180)
                tag = f"p{p_idx} S{stg} ({p_idx/12:.1f}s)"
            else:
                col = (0, 80, 255)
                tag = f"p{p_idx} NONE ({p_idx/12:.1f}s)"
            sm = cv2.resize(fr, (240, 320))
            cv2.putText(sm, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)
            tiles.append(sm)
    dec.stdout.close()
    dec.wait()

    cols = 6
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r : r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(VERIFY_IMG_PATH, np.vstack(rows))
    print(f"Wrote {VERIFY_IMG_PATH}")


if __name__ == "__main__":
    main()
