#!/usr/bin/env python3
"""Hybrid 12-fps planar screen tracker for the six centre-display glances.

Combines:
1. Direct per-pose 4-edge Huber line-intersection fitting on the un-cartoonized
   active LCD screen in work/final3_parts/head_raw.mp4 (7x7 close so bezel
   capacitive dots are never bridged into the screen contour).
2. Bidirectional LK optical-flow + RANSAC planar homography propagation across
   consecutive 12-fps poses (handling boundary-clipped poses in Panels 4 and 5
   and the bottom-left shadow in Panel 2).
3. 12-fps cartoon cadence lock: frames belonging to the same 12-fps drawing in
   work/final3_parts/head_cartoon.mp4 share the exact same quad (0.00 px
   within-pair jitter).
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import media_base as B

HEAD_RAW = os.path.join(B.WORK, "final3_parts", "head_raw.mp4")
HEAD_CARTOON = os.path.join(B.WORK, "final3_parts", "head_cartoon.mp4")

# Canonical intervals from v1 glance intervals
HEAD_SIGNS = [
    (1, 11.9, 12.6),
    (2, 91.15, 91.85),
    (3, 141.4, 141.85),
    (4, 204.2, 204.6),
    (5, 237.9, 238.15),
    (6, 240.65, 241.35),
]


def read_bgr(path):
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    n = 720 * 960 * 3
    frames = []
    while True:
        buf = dec.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(960, 720, 3).copy())
    dec.wait()
    return frames


def extract_raw_preroll(start, end, preroll_frames=2, postroll_frames=2):
    dt_pre = preroll_frames / 24.0
    dt_post = postroll_frames / 24.0
    s0 = max(0.0, start - dt_pre)
    actual_pre = start - s0
    dur = (end - start) + actual_pre + dt_post
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{s0:.6f}", "-i", HEAD_RAW, "-t", f"{dur:.6f}",
        "-vf", "setpts=N/(24*TB),fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 720 * 960 * 3
    frames = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(960, 720, 3).copy())
    proc.wait()
    return frames


def group_12fps_poses(cart_frames):
    """Group consecutive cartoon frames sharing the same 12-fps drawing."""
    groups = [[0]]
    for i in range(1, len(cart_frames)):
        diff = float(np.abs(cart_frames[i].astype(np.float32) - cart_frames[i - 1].astype(np.float32)).mean())
        if diff < 6.0:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def match_pose_to_raw(cart_frame, raw_frames, hint_idx):
    """Find the raw frame index matching a cartoon 12-fps pose via dashboard edge correlation."""
    cg = cv2.Canny(cv2.cvtColor(cart_frame[480:820, 260:720], cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
    cg_b = cv2.GaussianBlur(cg, (5, 5), 0)
    best_j, best_score = hint_idx, -1.0
    lo = max(0, hint_idx - 3)
    hi = min(len(raw_frames), hint_idx + 4)
    for j in range(lo, hi):
        rg = cv2.Canny(cv2.cvtColor(raw_frames[j][480:820, 260:720], cv2.COLOR_BGR2GRAY), 50, 150).astype(np.float32)
        score = float((cg_b * cv2.GaussianBlur(rg, (5, 5), 0)).mean())
        if score > best_score:
            best_score, best_j = score, j
    return best_j


def fit_line_2d(pts):
    vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    return np.array([vx, vy, x0, y0], dtype=np.float64)


def line_from_points(pa, pb):
    d = np.array(pb, np.float64) - np.array(pa, np.float64)
    n = np.hypot(d[0], d[1]) + 1e-12
    return np.array([d[0] / n, d[1] / n, pa[0], pa[1]], dtype=np.float64)


def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return None
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1], dtype=np.float64))[0]
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def detect_quad(fr, panel):
    """Detect active LCD quad and fitted edge lines on a single raw frame."""
    val = fr.max(axis=2)
    closed = cv2.morphologyEx(val, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    vthr = 115 if panel == 3 else (75 if panel == 2 else 65)
    th = (closed > vthr).astype(np.uint8) * 255
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (5000 < a < 22000):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if by < 490 or bx < 260 or by > 720 or bw < 100 or bw > 220 or bh < 55 or bh > 140:
            continue
        cands.append((a, bx, by, bw, bh, c))
    if not cands:
        return None, None
    _, bx, by, bw, bh, c = max(cands, key=lambda x: x[0])
    hull = cv2.convexHull(c)
    m = np.zeros(val.shape, np.uint8)
    cv2.drawContours(m, [hull], -1, 255, -1)

    xs = np.arange(int(bx + 0.18 * bw), int(bx + 0.82 * bw))
    top_pts, bot_pts = [], []
    for x in xs:
        col = np.where(m[:, x] > 0)[0]
        if len(col) > 20:
            top_pts.append([float(x), float(col[0])])
            bot_pts.append([float(x), float(col[-1])])

    ys = np.arange(int(by + 0.18 * bh), int(by + 0.82 * bh))
    left_pts, right_pts = [], []
    for y in ys:
        row = np.where(m[y, :] > 0)[0]
        if len(row) > 40:
            left_pts.append([float(row[0]), float(y)])
            right_pts.append([float(row[-1]), float(y)])

    if min(len(top_pts), len(bot_pts), len(left_pts), len(right_pts)) < 6:
        return None, None

    lt, lb = fit_line_2d(top_pts), fit_line_2d(bot_pts)
    ll, lr = fit_line_2d(left_pts), fit_line_2d(right_pts)
    tl = intersect_lines(ll, lt)
    tr = intersect_lines(lt, lr)
    br = intersect_lines(lr, lb)
    bl = intersect_lines(lb, ll)
    if any(p is None for p in (tl, tr, br, bl)):
        return None, None
    return np.array([tl, tr, br, bl], dtype=np.float32), (lt, lr, lb, ll)


def compute_lk_homography(f_src, f_dst, q_src):
    """Estimate planar homography from f_src to f_dst using forward-backward LK optical flow."""
    g0 = cv2.cvtColor(f_src, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f_dst, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(g0)
    x0 = max(0, int(q_src[:, 0].min()) - 35)
    y0 = max(0, int(q_src[:, 1].min()) - 25)
    x1 = min(720, int(q_src[:, 0].max()) + 35)
    y1 = min(960, int(q_src[:, 1].max()) + 25)
    mask[y0:y1, x0:x1] = 255
    pts0 = cv2.goodFeaturesToTrack(g0, 280, 0.006, 4, mask=mask)
    if pts0 is None or len(pts0) < 12:
        return None
    lk_kw = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **lk_kw)
    pts0_back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, pts1, None, **lk_kw)
    fb_err = np.hypot(*(pts0.reshape(-1, 2) - pts0_back.reshape(-1, 2)).T)
    ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb_err < 1.0)
    if ok.sum() < 10:
        ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb_err < 2.0)
    if ok.sum() < 8:
        return None
    H, inl = cv2.findHomography(pts0[ok], pts1[ok], cv2.RANSAC, 2.0)
    if H is None or inl is None or int(inl.sum()) < 8:
        return None
    return H


def warp_quad(q, H):
    return cv2.perspectiveTransform(q.reshape(1, -1, 2).astype(np.float32), H).reshape(-1, 2)


def pad_quad_outward(q, pad_tb, pad_lr):
    """Offset each edge line outward along its outward normal and re-intersect corners."""
    c = q.mean(axis=0)
    edges = [(0, 1, pad_tb), (1, 2, pad_lr), (2, 3, pad_tb), (3, 0, pad_lr)]
    lines = []
    for ia, ib, pad in edges:
        pa = q[ia].astype(np.float64)
        pb = q[ib].astype(np.float64)
        d = pb - pa
        d = d / (np.hypot(d[0], d[1]) + 1e-12)
        n = np.array([-d[1], d[0]], dtype=np.float64)
        mid = 0.5 * (pa + pb)
        if np.dot(n, mid - c) < 0:
            n = -n
        pa_s = pa + n * pad
        lines.append(np.array([d[0], d[1], pa_s[0], pa_s[1]], dtype=np.float64))
    lt, lr, lb, ll = lines
    return np.array(
        [intersect_lines(ll, lt), intersect_lines(lt, lr), intersect_lines(lr, lb), intersect_lines(lb, ll)],
        dtype=np.float32,
    )


def track_panel(panel, start, end, tmp_dir):
    # Extract cartoon segment using the exact same B.extract helper
    cart_seg = os.path.join(tmp_dir, f"cart_p{panel}.mp4")
    B.extract(HEAD_CARTOON, start, end - start, cart_seg)
    cart_frames = read_bgr(cart_seg)
    pose_groups = group_12fps_poses(cart_frames)

    # Extract raw frames with a 2-frame pre-roll so odd start boundaries (Panel 4) match cleanly
    pre_frames = 2
    raw_frames = extract_raw_preroll(start, end, preroll_frames=pre_frames, postroll_frames=2)

    pose_raw_idxs = []
    hint = pre_frames
    for grp in pose_groups:
        r_idx = match_pose_to_raw(cart_frames[grp[0]], raw_frames, hint)
        pose_raw_idxs.append(r_idx)
        hint = r_idx + 2

    # Direct per-pose detection
    det_quads = []
    det_lines = []
    for r_idx in pose_raw_idxs:
        q, lines = detect_quad(raw_frames[r_idx], panel)
        det_quads.append(q)
        det_lines.append(lines)

    n_poses = len(pose_groups)
    # Build forward and backward LK homographies between adjacent poses
    H_fwd = [None] * max(0, n_poses - 1)
    H_bwd = [None] * max(0, n_poses - 1)
    for k in range(n_poses - 1):
        q_ref = det_quads[k] if det_quads[k] is not None else det_quads[k + 1]
        r0, r1 = pose_raw_idxs[k], pose_raw_idxs[k + 1]
        H_fwd[k] = compute_lk_homography(raw_frames[r0], raw_frames[r1], q_ref)
        if det_quads[k + 1] is not None:
            H_bwd[k] = compute_lk_homography(raw_frames[r1], raw_frames[r0], det_quads[k + 1])

    # Handle boundary-clipped / shadow poses before smoothing
    poses = [q.copy() if q is not None else None for q in det_quads]

    if panel == 4 and poses[1] is not None and H_bwd[0] is not None:
        # Pose 0 has right side clipped at x=719; propagate right edge backward from pose 1
        q_prop = warp_quad(poses[1], H_bwd[0])
        if det_lines[0] is not None:
            lt0, _, lb0, ll0 = det_lines[0]
            lr_prop = line_from_points(q_prop[1], q_prop[2])
            tl = intersect_lines(ll0, lt0)
            tr = intersect_lines(lt0, lr_prop)
            br = intersect_lines(lr_prop, lb0)
            bl = intersect_lines(lb0, ll0)
            if all(p is not None for p in (tl, tr, br, bl)):
                poses[0] = np.array([tl, tr, br, bl], dtype=np.float32)
            else:
                poses[0] = q_prop
        else:
            poses[0] = q_prop

    if panel == 5 and n_poses >= 3 and poses[1] is not None and H_fwd[1] is not None:
        # Pose 2 (f04..05) approaches/crosses x=719; propagate right edge from pose 1
        q_prop = warp_quad(poses[1], H_fwd[1])
        if det_lines[2] is not None:
            lt2, _, lb2, ll2 = det_lines[2]
            lr_prop = line_from_points(q_prop[1], q_prop[2])
            tl = intersect_lines(ll2, lt2)
            tr = intersect_lines(lt2, lr_prop)
            br = intersect_lines(lr_prop, lb2)
            bl = intersect_lines(lb2, ll2)
            if all(p is not None for p in (tl, tr, br, bl)):
                poses[2] = np.array([tl, tr, br, bl], dtype=np.float32)
            else:
                poses[2] = q_prop
        else:
            poses[2] = q_prop

    if panel == 2 and n_poses >= 3 and poses[2] is not None and H_bwd[1] is not None and H_bwd[0] is not None:
        # Dark shadow at bottom-left of reversing camera on poses 0 and 1 can pull BL down by ~4px
        q1_bwd = warp_quad(poses[2], H_bwd[1])
        q0_bwd = warp_quad(q1_bwd, H_bwd[0])
        if poses[1] is not None:
            poses[1][3] = 0.25 * poses[1][3] + 0.75 * q1_bwd[3]
        if poses[0] is not None:
            poses[0][3] = 0.20 * poses[0][3] + 0.80 * q0_bwd[3]

    # Measure direct-fit vs LK-homography corner residuals prior to smoothing
    step_residuals = []
    for k in range(n_poses - 1):
        if H_fwd[k] is not None and poses[k] is not None and poses[k + 1] is not None:
            q_pred = warp_quad(poses[k], H_fwd[k])
            err = np.hypot(*(poses[k + 1] - q_pred).T)
            step_residuals.append(float(err.mean()))

    # Lightweight homography-consistent smoothing across adjacent 12-fps poses (zero phase lag)
    smoothed = [q.copy() for q in poses]
    for k in range(1, n_poses - 1):
        if H_fwd[k - 1] is not None and H_bwd[k] is not None:
            q_from_prev = warp_quad(poses[k - 1], H_fwd[k - 1])
            q_from_next = warp_quad(poses[k + 1], H_bwd[k])
            smoothed[k] = 0.75 * poses[k] + 0.25 * 0.5 * (q_from_prev + q_from_next)

    # Pad outward along edge normals to ensure cartoon bezel ink + LCD halo are fully covered
    if panel == 6:
        pad_tb, pad_lr = 3.5, 4.2
    elif panel == 2:
        pad_tb, pad_lr = 2.5, 3.0
    else:
        pad_tb, pad_lr = 2.0, 2.0

    final_poses = [pad_quad_outward(q, pad_tb, pad_lr) for q in smoothed]

    # Expand 12-fps pose quads to 24-fps segment frames (identical quad per 12-fps pose group)
    quads_24fps = [None] * len(cart_frames)
    for k, grp in enumerate(pose_groups):
        for f_idx in grp:
            quads_24fps[f_idx] = final_poses[k].tolist()

    within_pair_motion = []
    for grp in pose_groups:
        if len(grp) > 1:
            for j in range(1, len(grp)):
                d = np.max(np.hypot(*(np.array(quads_24fps[grp[j]]) - np.array(quads_24fps[grp[0]])).T))
                within_pair_motion.append(float(d))

    metrics = {
        "panel": panel,
        "frames": len(cart_frames),
        "unique_12fps_poses": n_poses,
        "pose_groups": pose_groups,
        "pose_raw_indices": pose_raw_idxs,
        "mean_interpose_homography_residual_px": round(float(np.mean(step_residuals)), 3) if step_residuals else 0.0,
        "max_interpose_homography_residual_px": round(float(np.max(step_residuals)), 3) if step_residuals else 0.0,
        "per_step_residuals_px": [round(r, 3) for r in step_residuals],
        "max_within_12fps_pair_motion_px": round(float(np.max(within_pair_motion)), 4) if within_pair_motion else 0.0,
    }
    return quads_24fps, cart_frames, metrics


def main():
    tmp_dir = os.path.join(B.WORK, "planar_parts")
    diag_dir = os.path.join(B.WORK, "track_diag")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    out_quads = {}
    all_metrics = {}
    for panel, start, end in HEAD_SIGNS:
        quads, cart_frames, metrics = track_panel(panel, start, end, tmp_dir)
        out_quads[str(panel)] = dict(start=start, fps=24.0, quads=quads)
        all_metrics[str(panel)] = metrics
        print(
            f"Panel {panel}: {metrics['frames']} frames ({metrics['unique_12fps_poses']} poses), "
            f"mean residual={metrics['mean_interpose_homography_residual_px']:.2f} px, "
            f"max residual={metrics['max_interpose_homography_residual_px']:.2f} px, "
            f"within-pair drift={metrics['max_within_12fps_pair_motion_px']:.3f} px"
        )

        # Save start/mid/end quad overlay diagnostic
        imgs = []
        for idx in (0, len(cart_frames) // 2, len(cart_frames) - 1):
            f = cart_frames[idx].copy()
            cv2.polylines(f, [np.round(quads[idx]).astype(np.int32)], True, (0, 255, 0), 2)
            imgs.append(cv2.resize(f, (360, 480)))
        cv2.imwrite(os.path.join(diag_dir, f"panel{panel}_quads.jpg"), np.hstack(imgs))

    quads_path = os.path.join(B.WORK, "track_quads.json")
    with open(quads_path, "w") as f:
        json.dump(out_quads, f, indent=2)
    metrics_path = os.path.join(diag_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Wrote", quads_path, "and", metrics_path)


if __name__ == "__main__":
    main()
