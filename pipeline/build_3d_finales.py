#!/usr/bin/env python3
"""Render non-destructive 3D-perspective themed-ending versions of the three films:
  - output/01_nansen_into-the-white-silence_themed-ending.mp4
  - output/02_shackleton_endurance-in-the-dark_themed-ending.mp4
  - output/03_mars500_520-days-to-another-world_themed-ending.mp4

Eliminates 100% of Carlos's reflections, car-door exit, garage walk, house-key
insertion, and modern elevator doorway reflection (poses >= 2989 of head_raw.mp4
and all of tail_raw.mp4):
  1. Preserves every drive frame cleanly up to the car halt (f_stop - 1).
  2. Dissolves into a 12-fps 6-DoF pinhole-perspective walk through three layered
     3D scenes:
       - Arr1: Outer doorway/hatch surround (Z=1.0) expands off-screen as the
         camera steps through into the exterior 3D world.
       - Arr2: Depth-layered corridor with separated foreground props (sled,
         tents/lifeboat, tunnel ribs) & portal-locked arrival into Arr3.
       - Arr3: Clean symmetrical door + dynamic 3D reaching hand that presses/
         turns the latch and retracts completely before opening.
  3. Dollies through the outer architectural archway (Z=1.0) while recessed door
     leaves (Z=1.15) part behind the frame into a 3D concave protagonist cabin
     + Outro legacy card.

Usage:
  uv run --with opencv-python-headless --with numpy --with scipy --with pillow \
      python3 -m pipeline.build_3d_finales preview
  uv run --with opencv-python-headless --with numpy --with scipy --with pillow \
      python3 -m pipeline.build_3d_finales all
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import build_trilogy_films as BTF, build_trilogy_tails as BTC, cubist_portrait as CC, media_base as B, night_chalk_shader as R

W, H = R.W, R.H
FOCAL = 660.0
ARRIVAL_DIR = os.path.join(B.WORK, "trilogy_assets", "arrival")
TRI_DIR = BTF.TRI_DIR
DIAG_DIR = BTF.DIAG_DIR
CAR_STOP_POSE = 2989

SEAM_X = {
    "nansen": 385,
    "shackleton": 359,
    "mars500": 359,
}

# Portal interior box inside clean Arr3 where recessed doors slide behind outer archway:
# (x_left, x_right, y_top, door_bottom)
ARCH_BOUNDS = {
    "nansen": (184, 575, 148, H),
    "shackleton": (182, 594, 130, H),
    "mars500": (172, 546, 0, H),
}


def _rot_matrix(pitch_deg, yaw_deg, roll_deg):
    """3D camera rotation matrix R_wc = R_y(yaw) @ R_x(pitch) @ R_z(roll)."""
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    r = np.deg2rad(roll_deg)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return Ry @ Rx @ Rz


def _project_3d_layer(
    img_bgr,
    z_map,
    vp,
    tx=0.0,
    ty=0.0,
    tz=0.0,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
    focal=FOCAL,
    alpha=None,
    border_const=False,
):
    """Project a textured depth layer through a 6-DoF pinhole camera using 2-step ray intersection."""
    vx, vy = vp
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rx = (xx - vx) / focal
    ry = (yy - vy) / focal
    rz = np.ones_like(rx)

    R_wc = _rot_matrix(pitch, yaw, roll)
    dx = R_wc[0, 0] * rx + R_wc[0, 1] * ry + R_wc[0, 2] * rz
    dy = R_wc[1, 0] * rx + R_wc[1, 1] * ry + R_wc[1, 2] * rz
    dz = np.maximum(0.18, R_wc[2, 0] * rx + R_wc[2, 1] * ry + R_wc[2, 2] * rz)

    Z_est = z_map.astype(np.float32)
    z_f32 = Z_est
    for _ in range(2):
        # Distance along camera ray before depth collapse clamp
        dz_hit = np.maximum(0.08, Z_est - tz)
        s = dz_hit / dz
        X = tx + s * dx
        Y = ty + s * dy
        xs = (vx + focal * (X / Z_est)).astype(np.float32)
        ys = (vy + focal * (Y / Z_est)).astype(np.float32)
        Z_sampled = cv2.remap(
            z_f32,
            xs,
            ys,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        Z_est = (0.35 * Z_est + 0.65 * Z_sampled).astype(np.float32)

    dz_hit = np.maximum(0.08, Z_est - tz)
    s = dz_hit / dz
    xs = (vx + focal * ((tx + s * dx) / Z_est)).astype(np.float32)
    ys = (vy + focal * ((ty + s * dy) / Z_est)).astype(np.float32)

    bmode = cv2.BORDER_CONSTANT if border_const else cv2.BORDER_REFLECT101
    warped = cv2.remap(img_bgr, xs, ys, cv2.INTER_LINEAR, borderMode=bmode, borderValue=(0, 0, 0))
    if alpha is None:
        return warped, None

    # Clip cleanly if portal expands past camera near plane (Z <= tz + 0.08)
    valid_near = (Z_est - tz > 0.12).astype(np.float32)
    w_alpha = cv2.remap(
        alpha.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    return warped, np.clip(w_alpha * valid_near, 0.0, 1.0)


def _walk_kinematics(t_walk, step_hz, amp=1.0):
    """Natural human first-person gait bounce + coupled head pitch nod + stride sway."""
    ph = np.pi * step_hz * t_walk
    # Double-frequency vertical heel-strike impulse minus average offset
    heel = (1.0 - np.abs(np.cos(ph))) * 2.0 - 1.0
    ty = 0.022 * amp * heel
    tx = 0.018 * amp * np.sin(ph)
    pitch = 0.55 * amp * (-heel) + 0.35 * amp * np.sin(2.0 * np.pi * t_walk * 0.22)
    yaw = 1.15 * amp * np.sin(ph * 0.5 + 0.4) + 0.85 * amp * np.sin(2.0 * np.pi * t_walk * 0.18)
    roll = 0.48 * amp * np.sin(ph)
    return tx, ty, pitch, yaw, roll


def _prepare_arr1_layers(bgr, key):
    """Separate Arr1 foreground doorway/hatch frame (Z=0.92) from exterior 3D depth world."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    portal_mask = np.ones((H, W), dtype=np.float32)

    if key == "nansen":
        vp = (360.0, 410.0)
        # Interior ice deck & Fram opening: [112..608, 128..960]
        portal_mask[128:H, 112:608] = 0.0
        portal_mask = cv2.GaussianBlur(portal_mask, (17, 17), 4.0)
    elif key == "shackleton":
        vp = (355.0, 430.0)
        portal_mask[122:855, 118:585] = 0.0
        portal_mask = cv2.GaussianBlur(portal_mask, (17, 17), 4.0)
    else:
        vp = (485.0, 620.0)
        poly = np.array([[85, 540], [125, 310], [355, 72], [542, 72], [618, 225], [600, 959], [95, 959]], np.int32)
        cv2.fillPoly(portal_mask, [poly], 0.0)
        portal_mask = cv2.GaussianBlur(portal_mask, (15, 15), 3.5)

    # Bounded continuous exterior depth: Z=1.12 at y=H rising smoothly to Z=1.72 at y<=vp[1]
    t_sky = np.clip((H - yy) / max(1.0, float(H - vp[1])), 0.0, 1.0)
    z_world = 1.12 + 0.60 * (t_sky ** 0.90)
    z_world = cv2.GaussianBlur(z_world.astype(np.float32), (31, 31), 9.0)

    return dict(
        vp=vp,
        portal_bgr=bgr.copy(),
        portal_alpha=portal_mask,
        portal_z=np.full((H, W), 0.92, dtype=np.float32),
        world_bgr=bgr.copy(),
        world_z=z_world,
    )


def _prepare_arr2_layers(bgr, key):
    """Build smooth continuous 3D corridor + foreground prop relief depth map (zero TELEA smudge)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    if key == "nansen":
        vp = (370.0, 488.0)
        door_rect = (305, 392, 435, 588)
        rx = (xx - vp[0]) / 260.0
        ry = np.where(yy > vp[1], (yy - vp[1]) / 360.0, (vp[1] - yy) / 420.0)
        rad = np.maximum(np.abs(rx), ry)
        z_corr = np.clip(1.25 / np.maximum(0.24, rad ** 0.78), 1.22, 5.0)
        # Smooth foreground relief island on left dog sled (brings sled slightly closer than snow behind it)
        prop_m = np.zeros((H, W), dtype=np.float32)
        sled_poly = np.array([[18, 790], [62, 565], [160, 545], [312, 625], [310, 745], [135, 862], [18, 862]], np.int32)
        cv2.fillPoly(prop_m, [sled_poly], 1.0)
        prop_m = cv2.GaussianBlur(prop_m, (25, 25), 7.0)
        z_corr = z_corr * (1.0 - 0.24 * prop_m)
    elif key == "shackleton":
        vp = (372.0, 495.0)
        door_rect = (312, 395, 438, 595)
        rx = (xx - vp[0]) / 270.0
        ry = np.where(yy > vp[1], (yy - vp[1]) / 370.0, (vp[1] - yy) / 430.0)
        rad = np.maximum(np.abs(rx), ry)
        z_corr = np.clip(1.25 / np.maximum(0.24, rad ** 0.78), 1.22, 5.0)
        prop_m = np.zeros((H, W), dtype=np.float32)
        tent_poly = np.array([[0, 505], [125, 440], [285, 650], [210, 720], [0, 720]], np.int32)
        boat_poly = np.array([[432, 615], [535, 495], [719, 470], [719, 825], [505, 815]], np.int32)
        cv2.fillPoly(prop_m, [tent_poly], 1.0)
        cv2.fillPoly(prop_m, [boat_poly], 1.0)
        prop_m = cv2.GaussianBlur(prop_m, (25, 25), 7.0)
        z_corr = z_corr * (1.0 - 0.24 * prop_m)
    else:
        vp = (365.0, 502.0)
        door_rect = (280, 418, 450, 588)
        r_norm = np.hypot((xx - vp[0]) / 260.0, (yy - vp[1]) / 330.0)
        z_corr = np.clip(1.20 / np.maximum(0.24, r_norm ** 0.85), 1.20, 4.9)

    z_corr = cv2.GaussianBlur(z_corr.astype(np.float32), (25, 25), 6.0)
    x0, y0, x1, y1 = door_rect
    z_corr[y0:y1, x0:x1] = np.maximum(z_corr[y0:y1, x0:x1], 4.2)

    return dict(
        vp=vp,
        door_rect=door_rect,
        bg_bgr=bgr.copy(),
        bg_z=z_corr,
    )


def _prepare_arr3_layers(bgr, key):
    """Extract clean symmetrical hand-free door, foreground reaching hand matte, and split archway/door layers."""
    src = bgr.astype(np.float32)
    clean = bgr.copy().astype(np.float32)

    # --- 1. Clean symmetrical hand-free door reconstruction ---
    if key == "nansen":
        cx, hw = 385, 190
        left_flip = src[:, cx - hw : cx][:, ::-1].copy()
        l_col = left_flip[200:540].mean(axis=0, keepdims=True) + 1e-6
        r_col = src[200:540, cx : cx + hw].mean(axis=0, keepdims=True) + 1e-6
        right_clean = np.clip(left_flip * np.clip(r_col / l_col, 0.80, 1.25), 0, 255)

        # Clone clean upper-right icy stone border downward (avoiding duplicating left call button)
        rock_tile = src[180:565, cx + hw : W].copy()
        rock_gain = (src[565:950, 0:90].mean() + 1e-6) / (src[180:565, 0:90].mean() + 1e-6)
        rock_lower = np.clip(rock_tile * np.clip(rock_gain, 0.80, 1.15), 0, 255)

        synth = clean.copy()
        synth[:, cx : cx + hw] = right_clean
        synth[575:960, cx + hw : W] = rock_lower

        m = np.zeros((H, W), dtype=np.float32)
        poly = np.array([[435, 585], [510, 562], [625, 570], [719, 670], [719, 959], [540, 959], [435, 710]], np.int32)
        cv2.fillPoly(m, [poly], 1.0)
        m = cv2.dilate(m, np.ones((7, 7), np.uint8))
        m = cv2.GaussianBlur(m, (11, 11), 2.5)[:, :, None]
        clean = clean * (1.0 - m) + synth * m

    elif key == "shackleton":
        cx, hw = 359, 232
        left_flip = src[:, cx - hw : cx][:, ::-1].copy()
        l_col = left_flip[320:540].mean(axis=0, keepdims=True) + 1e-6
        r_col = src[320:540, cx : cx + hw].mean(axis=0, keepdims=True) + 1e-6
        right_planks = np.clip(left_flip * np.clip(r_col / l_col, 0.75, 1.30), 0, 255)

        y_src0, span = 275, 335
        tile = src[y_src0 : y_src0 + span, :].copy()
        l_up = src[y_src0 : y_src0 + span, 220:330].mean(axis=(1, 2), keepdims=True) + 1e-6
        l_dn = src[625:960, 220:330].mean(axis=(1, 2), keepdims=True)
        planks_lower = np.clip(tile * np.clip(l_dn / l_up, 0.58, 1.12), 0, 255)

        synth = clean.copy()
        synth[560:960, cx : cx + hw] = right_planks[560:960]
        synth[625:960, cx + hw : W] = planks_lower[:, cx + hw : W]

        m = np.zeros((H, W), dtype=np.float32)
        poly = np.array([[420, 622], [625, 620], [719, 705], [719, 959], [420, 959]], np.int32)
        cv2.fillPoly(m, [poly], 1.0)
        m = cv2.dilate(m, np.ones((9, 9), np.uint8))
        m = cv2.GaussianBlur(m, (11, 11), 2.5)[:, :, None]
        clean = clean * (1.0 - m) + synth * m

    else:
        cx, hw = 359, 187
        left_flip = src[:, cx - hw : cx][:, ::-1].copy()
        l_col = left_flip[150:550].mean(axis=0, keepdims=True) + 1e-6
        r_col = src[150:550, cx : cx + hw].mean(axis=0, keepdims=True) + 1e-6
        right_clean = np.clip(left_flip * np.clip(r_col / l_col, 0.82, 1.22), 0, 255)

        outer_flip = src[:, :174][:, ::-1].copy()
        synth = clean.copy()
        synth[:, cx : cx + hw] = right_clean
        synth[:, cx + hw : W] = outer_flip[:, : W - (cx + hw)]

        m = np.zeros((H, W), dtype=np.float32)
        poly = np.array([[378, 608], [492, 608], [625, 705], [719, 760], [719, 959], [450, 959], [375, 755]], np.int32)
        cv2.fillPoly(m, [poly], 1.0)
        # Preserve only the clean upper-right orange toggle untouched above sleeve
        sw_keep = np.zeros((H, W), dtype=np.float32)
        sw_keep[676:752, 648:704] = 1.0
        m = np.clip(m - cv2.GaussianBlur(sw_keep, (7, 7), 1.2), 0.0, 1.0)
        m = cv2.dilate(m, np.ones((7, 7), np.uint8))
        m = cv2.GaussianBlur(m, (9, 9), 2.0)[:, :, None]
        clean = clean * (1.0 - m) + synth * m

    clean_u8 = np.clip(clean, 0, 255).astype(np.uint8)

    # --- 2. GrabCut reaching hand matte ---
    gc = np.full((H, W), cv2.GC_BGD, dtype=np.uint8)
    if key == "nansen":
        poly = np.array([[435, 610], [495, 575], [598, 588], [692, 678], [719, 770], [719, 959], [565, 959], [505, 765], [435, 690]], np.int32)
        cv2.fillPoly(gc, [poly], cv2.GC_PR_FGD)
        cv2.line(gc, (455, 635), (555, 645), cv2.GC_FGD, 24)
        cv2.circle(gc, (535, 660), 30, cv2.GC_FGD, -1)
        cv2.line(gc, (560, 735), (678, 765), cv2.GC_FGD, 38)
        cv2.line(gc, (625, 815), (705, 935), cv2.GC_FGD, 50)
    elif key == "shackleton":
        poly = np.array([[488, 635], [568, 618], [660, 690], [719, 780], [719, 959], [598, 959], [542, 795], [488, 745]], np.int32)
        cv2.fillPoly(gc, [poly], cv2.GC_PR_FGD)
        cv2.line(gc, (515, 655), (540, 745), cv2.GC_FGD, 22)
        cv2.circle(gc, (560, 700), 28, cv2.GC_FGD, -1)
        cv2.line(gc, (585, 755), (695, 930), cv2.GC_FGD, 42)
    else:
        poly = np.array([[378, 730], [412, 622], [488, 608], [588, 695], [715, 855], [719, 959], [510, 959], [430, 810]], np.int32)
        cv2.fillPoly(gc, [poly], cv2.GC_PR_FGD)
        cv2.line(gc, (428, 645), (500, 730), cv2.GC_FGD, 24)
        cv2.line(gc, (462, 625), (528, 710), cv2.GC_FGD, 24)
        cv2.circle(gc, (520, 760), 42, cv2.GC_FGD, -1)
        cv2.rectangle(gc, (578, 822), (648, 900), cv2.GC_FGD, -1)
        cv2.line(gc, (625, 895), (685, 950), cv2.GC_FGD, 44)
        b, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 2].astype(np.int16)
        gc[615:655, 405:452] = np.where((b[615:655, 405:452] - r[615:655, 405:452]) < 15, cv2.GC_BGD, gc[615:655, 405:452])

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros_like(fg)
    if num > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep[lbl == best] = 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    hand_alpha = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (5, 5), 1.0)

    # Archway mask: outside portal opening [x_l..x_r, y_t..y_b]
    x_l, x_r, y_t, y_b = ARCH_BOUNDS[key]
    arch_alpha = np.ones((H, W), dtype=np.float32)
    arch_alpha[y_t:y_b, x_l:x_r] = 0.0
    arch_alpha = cv2.GaussianBlur(arch_alpha, (5, 5), 1.0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vp = (float(SEAM_X[key]), 480.0)
    # Recessed door is slightly further back (Z=1.16) than stone/wood/steel archway surround (Z=1.00)
    z_arch = np.full((H, W), 1.00, dtype=np.float32)
    z_arch[y_t:y_b, x_l:x_r] = 1.15
    # Slight relief bevel on archway edge
    z_arch = cv2.GaussianBlur(z_arch, (21, 21), 5.0)

    return dict(
        vp=vp,
        clean_bgr=clean_u8,
        clean_rgb=clean_u8[:, :, [2, 1, 0]],
        hand_bgr=bgr.copy(),
        hand_alpha=hand_alpha,
        arch_alpha=arch_alpha,
        z_map=z_arch,
        seam_x=SEAM_X[key],
        arch_bounds=ARCH_BOUNDS[key],
    )


def _load_arrival_layers(key):
    layers = []
    for idx in range(1, 4):
        p = os.path.join(ARRIVAL_DIR, f"{key}_arr{idx}.png")
        rgb = np.array(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS))
        bgr = rgb[:, :, [2, 1, 0]].copy()
        if idx == 1:
            layers.append(_prepare_arr1_layers(bgr, key))
        elif idx == 2:
            layers.append(_prepare_arr2_layers(bgr, key))
        else:
            layers.append(_prepare_arr3_layers(bgr, key))
    return layers


def _render_arr1_3d(l1, sc_u, t_walk):
    """Render 3D portal exit walk through Arr1 frame into open exterior world."""
    tx, ty, pitch, yaw, roll = _walk_kinematics(t_walk, step_hz=1.42, amp=1.0)
    tz = 0.58 * float(R.smoothstep(0.0, 1.0, sc_u))
    world_w, _ = _project_3d_layer(l1["world_bgr"], l1["world_z"], l1["vp"], tx, ty, tz, pitch, yaw, roll)
    port_w, a_w = _project_3d_layer(
        l1["portal_bgr"], l1["portal_z"], l1["vp"], tx, ty, tz, pitch, yaw, roll,
        alpha=l1["portal_alpha"], border_const=True
    )
    comp = world_w.astype(np.float32) * (1.0 - a_w[:, :, None]) + port_w.astype(np.float32) * a_w[:, :, None]
    return np.clip(comp, 0, 255).astype(np.uint8)


def _render_arr2_3d(l2, l3, sc_u, t_walk, portal_u=0.0):
    """Render 3D corridor walk with continuous relief depth parallax + portal-aligned Arr3 zoom-in."""
    settle = 1.0 - float(R.smoothstep(0.76, 1.0, sc_u))
    tx, ty, pitch, yaw, roll = _walk_kinematics(t_walk, step_hz=1.50, amp=0.95 * (0.25 + 0.75 * settle))
    tz = 0.72 * float(R.smoothstep(0.0, 1.0, sc_u))
    yaw *= settle

    comp_u8, _ = _project_3d_layer(l2["bg_bgr"], l2["bg_z"], l2["vp"], tx, ty, tz, pitch, yaw, roll)
    comp = comp_u8.astype(np.float32)

    # Portal-locked 3D transition into Arr3 clean door
    if portal_u > 0.001:
        e = float(R.smoothstep(0.0, 1.0, portal_u))
        # True 3D camera pullback tz < 0 shrinks Arr3 inside Arr2's distant doorway and expands smoothly to 1.0x
        tz_arr3 = -2.15 * (1.0 - e)
        # Soft feather on outer edges of Arr3 panel while expanding out of doorway
        edge_a = np.ones((H, W), dtype=np.float32)
        edge_a[:, :14] = np.linspace(0.0, 1.0, 14, dtype=np.float32)[None, :]
        edge_a[:, -14:] = np.linspace(1.0, 0.0, 14, dtype=np.float32)[None, :]
        edge_a[:14, :] *= np.linspace(0.0, 1.0, 14, dtype=np.float32)[:, None]
        arr3_3d, a_arr3 = _project_3d_layer(
            l3["clean_bgr"],
            l3["z_map"],
            l3["vp"],
            tx * 0.35 * (1.0 - e),
            ty * 0.35 * (1.0 - e),
            tz_arr3,
            pitch * (1.0 - e),
            yaw * (1.0 - e),
            roll * (1.0 - e),
            alpha=edge_a,
            border_const=True,
        )
        alpha_blend = a_arr3 * float(R.smoothstep(0.0, 0.45, e))
        comp = comp * (1.0 - alpha_blend[:, :, None]) + arr3_3d.astype(np.float32) * alpha_blend[:, :, None]

    return np.clip(comp, 0, 255).astype(np.uint8)


def _render_arr3_3d(l3, sc_u, t_walk):
    """Render 3D approach to clean door + dynamic 3D foreground hand reach + full retraction."""
    settle = 1.0 - float(R.smoothstep(0.35, 0.90, sc_u))
    tx, ty, pitch, yaw, roll = _walk_kinematics(t_walk, step_hz=1.20, amp=0.55 * settle)
    tz = -0.08 * (1.0 - float(R.smoothstep(0.0, 0.55, sc_u))) + 0.04 * float(R.smoothstep(0.55, 0.95, sc_u))

    door_w, _ = _project_3d_layer(l3["clean_bgr"], l3["z_map"], l3["vp"], tx, ty, tz, pitch, yaw, roll)
    comp = door_w.astype(np.float32)

    # Dynamic 3D hand choreography inside sc_u in [0.16..0.98]: rises in, turns/presses, retracts down cleanly!
    if 0.14 < sc_u < 0.98:
        reach_in = float(R.smoothstep(0.14, 0.44, sc_u))
        retract_out = float(R.smoothstep(0.68, 0.96, sc_u))
        presence = reach_in * (1.0 - retract_out)
        # Slide hand upward from below frame when entering/retracting + 3D zoom & press twist at peak
        y_slide = (1.0 - presence) * 340.0
        x_slide = (1.0 - presence) * 95.0
        press_pulse = np.sin(np.pi * float(np.clip((sc_u - 0.42) / 0.28, 0.0, 1.0))) if 0.42 <= sc_u <= 0.70 else 0.0
        twist_deg = -4.5 * press_pulse
        # Affine hand offset in screen coords before slight camera follow
        M_h = cv2.getRotationMatrix2D((560.0, 740.0), twist_deg, 1.00 + 0.04 * press_pulse).astype(np.float32)
        M_h[0, 2] += x_slide - 10.0 * press_pulse + tx * 240.0
        M_h[1, 2] += y_slide + 6.0 * press_pulse + ty * 240.0

        hw_bgr = cv2.warpAffine(l3["hand_bgr"], M_h, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        hw_a = cv2.warpAffine(l3["hand_alpha"], M_h, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        # Drop shadow slightly down-left from reaching arm onto door behind
        sh_M = M_h.copy()
        sh_M[0, 2] -= 12.0
        sh_M[1, 2] += 10.0
        sh_a = cv2.GaussianBlur(
            cv2.warpAffine(l3["hand_alpha"], sh_M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
            (21, 21), 6.0
        ) * 0.42 * presence
        comp *= (1.0 - sh_a[:, :, None])
        comp = comp * (1.0 - hw_a[:, :, None]) + hw_bgr.astype(np.float32) * hw_a[:, :, None]

    return np.clip(comp, 0, 255).astype(np.uint8)


def _add_themed_atmosphere(f32, style_key, step_idx, t_sec, snow_bank, xx, yy):
    """Inject subtle 12-fps windblown arctic snow, warm lantern flicker, or Mars light shift."""
    if style_key == "nansen":
        fl = snow_bank[step_idx % len(snow_bank)]
        aur_w = 0.5 + 0.5 * np.sin(0.011 * xx - 0.007 * yy + 2.0 * np.pi * t_sec / 4.2)
        sky_w = (1.0 - R.smoothstep(0.32 * H, 0.62 * H, yy)) * aur_w
        f32 += sky_w[:, :, None] * np.array([24.0, 28.0, 6.0], dtype=np.float32) * 0.45
        f32 += fl[:, :, None] * np.array([240.0, 250.0, 225.0], dtype=np.float32) * 0.34
    elif style_key == "shackleton":
        fl = snow_bank[step_idx % len(snow_bank)]
        flicker = 0.7 + 0.3 * np.sin(2.0 * np.pi * t_sec / 2.3 + 0.6 * np.sin(2.0 * np.pi * t_sec / 1.1))
        ceil_w = 1.0 - R.smoothstep(0.20 * H, 0.58 * H, yy)
        f32 += (ceil_w * flicker)[:, :, None] * np.array([8.0, 18.0, 32.0], dtype=np.float32) * 0.42
        f32 += fl[:, :, None] * np.array([155.0, 195.0, 235.0], dtype=np.float32) * 0.24
    else:
        warm_mix = float(R.smoothstep(195.0, 232.0, t_sec))
        glow_col = np.array([255.0, 225.0, 95.0], dtype=np.float32) * (1.0 - warm_mix) + np.array([135.0, 210.0, 255.0], dtype=np.float32) * warm_mix
        pulse = 0.6 + 0.4 * np.sin(2.0 * np.pi * t_sec / 3.4)
        edge_mask = R.smoothstep(0.18 * H, 0.45 * H, yy) * (1.0 - R.smoothstep(0.55 * H, 0.88 * H, yy))
        f32 += (edge_mask * pulse)[:, :, None] * glow_col * 0.08
    return np.clip(f32, 0.0, 255.0)


def _build_snow_bank(n_variants=16):
    rng = np.random.default_rng(108)
    ker = R.make_line_kernel(9, -25.0)
    bank = []
    for k in range(n_variants):
        pts = np.zeros((H, W), dtype=np.float32)
        ys = rng.integers(0, H, size=140)
        xs = rng.integers(0, W, size=140)
        vals = rng.uniform(0.35, 1.0, size=140)
        pts[ys, xs] = vals
        pts = np.roll(np.roll(pts, k * 7, axis=0), k * 11, axis=1)
        streaks = cv2.filter2D(pts, -1, ker)
        streaks = cv2.GaussianBlur(streaks, (0, 0), 0.7)
        mx = float(streaks.max()) + 1e-6
        bank.append(np.clip(streaks / (mx * 0.55), 0.0, 1.0).astype(np.float32))
    return bank


def _render_arrival_frame_3d(layers, style_key, u, t_out, step_idx, shader, snow_bank, xx, yy):
    """Render one 12-fps stylized frame across the 3D arrival journey (Arr1 -> Arr2 -> Arr3)."""
    l1, l2, l3 = layers
    t_walk = step_idx / 12.0

    if u < 0.29:
        sc_u = float(np.clip(u / 0.34, 0.0, 1.0))
        raw_bgr = _render_arr1_3d(l1, sc_u, t_walk)
    elif u < 0.34:
        w1 = float(R.smoothstep(0.29, 0.34, u))
        f1 = _render_arr1_3d(l1, float(np.clip(u / 0.34, 0.0, 1.0)), t_walk)
        f2 = _render_arr2_3d(l2, l3, float(np.clip((u - 0.29) / 0.42, 0.0, 1.0)), t_walk, portal_u=0.0)
        raw_bgr = np.clip(f1.astype(np.float32) * (1.0 - w1) + f2.astype(np.float32) * w1, 0, 255).astype(np.uint8)
    elif u < 0.71:
        sc_u = float(np.clip((u - 0.29) / 0.42, 0.0, 1.0))
        # Smooth portal zoom from distant Arr2 door straight onto full-frame clean Arr3 [u in 0.54..0.71]
        portal_u = float(np.clip((u - 0.54) / 0.17, 0.0, 1.0))
        raw_bgr = _render_arr2_3d(l2, l3, sc_u, t_walk, portal_u=portal_u)
    else:
        sc_u = float(np.clip((u - 0.71) / 0.29, 0.0, 1.0))
        raw_bgr = _render_arr3_3d(l3, sc_u, t_walk)

    base = shader.prepare_base(raw_bgr)
    styled = shader.render_style(base, step_idx, style_key, t_sec=t_out)
    return _add_themed_atmosphere(styled, style_key, step_idx, t_out, snow_bank, xx, yy)


def _render_themed_head(vcfg, sched, f_stop, shader, layers, snow_bank, xx, yy):
    """Copy head_key[0..f_stop-1] verbatim and append 12-fps 3D arrival for f_stop..n_out-1."""
    k = vcfg["key"]
    style = vcfg["style"]
    n_out = len(sched)
    n_rem = n_out - f_stop

    # Non-destructive source: prefer original drive head_k.mp4 (not previously overwritten themed file)
    head_prev = os.path.join(TRI_DIR, f"head_{k}.mp4")
    head_out = os.path.join(TRI_DIR, f"head_{k}_themed.mp4")

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", head_prev, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "14", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-r", "24", head_out, "-y",
        ],
        stdin=subprocess.PIPE,
    )

    frame_bytes = W * H * 3
    last_parked_f32 = None
    for idx in range(f_stop):
        buf = dec.stdout.read(frame_bytes)
        assert len(buf) == frame_bytes, f"{k}: short read from {head_prev} at frame {idx}/{f_stop}"
        enc.stdin.write(buf)
        if idx == f_stop - 1:
            last_parked_f32 = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).astype(np.float32)

    dec.stdout.close()
    dec.wait()

    rng = np.random.default_rng(313 + sum(ord(c) for c in k))
    cached_f32 = None
    for j in range(n_rem):
        t_out = (f_stop + j) / 24.0
        u = j / float(max(1, n_rem - 1))
        step_idx = j // 2
        if j % 2 == 0 or cached_f32 is None:
            cached_f32 = _render_arrival_frame_3d(layers, style, u, t_out, step_idx, shader, snow_bank, xx, yy)

        out_f32 = cached_f32.copy()
        if j < 18 and last_parked_f32 is not None:
            w_arr = float(R.smoothstep(0.0, 18.0, j))
            out_f32 = last_parked_f32 * (1.0 - w_arr) + out_f32 * w_arr

        out_f32 += rng.normal(0.0, 2.2, (H, W, 1)).astype(np.float32)
        enc.stdin.write(np.clip(out_f32, 0.0, 255.0).astype(np.uint8).tobytes())

    enc.stdin.close()
    enc.wait()
    print(f"3D Themed head done ({k}): copied={f_stop}, arrival_3d={n_rem} -> {head_out}")
    return head_out


def _build_cubist_door_plate(l3, style_key, head_dur, shader, hatch):
    """Stylize clean hand-free Arr3 as a Cubist Delaunay canvas aligned 1:1 with end of head."""
    styled_bgr = np.clip(shader.render_style(shader.prepare_base(l3["clean_bgr"]), 0, style_key, t_sec=head_dur), 0, 255).astype(np.uint8)
    styled_rgb = styled_bgr[:, :, [2, 1, 0]]

    gray = cv2.cvtColor(styled_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tri = CC.make_points(gray, spacing=125, seed=41)
    tri_cols = CC.sample_triangle_colors(styled_rgb, tri)
    ramp_cfg = BTC.CUBIST_RAMPS[style_key]
    chosen = BTC._map_colors_ramp(tri_cols, ramp_cfg["WARM"], ramp_cfg["NEUT"], ramp_cfg["COOL"])
    fill_rgb = np.clip(0.50 * chosen + 0.50 * tri_cols, 0, 255).astype(int)

    img = Image.new("RGB", (W, H), ramp_cfg["ink"])
    dr = ImageDraw.Draw(img)
    pts, simps = tri.points, tri.simplices
    ink_rgb = ramp_cfg["ink"]
    for s, c in zip(simps, fill_rgb):
        dr.polygon([tuple(pts[idx]) for idx in s], fill=tuple(c), outline=ink_rgb, width=1)
    facet_bgr = np.asarray(img)[:, :, [2, 1, 0]].astype(np.float32)

    door_bgr = 0.64 * styled_bgr.astype(np.float32) + 0.36 * facet_bgr
    dog = cv2.GaussianBlur(gray, (0, 0), 0.75) - cv2.GaussianBlur(gray, (0, 0), 2.0)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) * 0.25
    ink = np.clip((np.clip((grad - 16.0) / 48.0, 0.0, 1.0) * 0.75 + np.clip((-dog - 2.0) / 7.5, 0.0, 1.0) * 0.62), 0.0, 0.80)[:, :, None]
    ink_col_bgr = np.array([ink_rgb[2], ink_rgb[1], ink_rgb[0]], dtype=np.float32)
    door_bgr = door_bgr * (1.0 - ink) + ink_col_bgr * ink
    door_bgr *= (1.0 + (0.045 * hatch)[:, :, None])
    return np.clip(door_bgr, 0.0, 255.0).astype(np.uint8)


def _build_cabin_depth(canvas_bgr, bust_mask):
    """Construct concave 3D elevator cabin depth map: foreground side walls & ceiling (Z=1.25) ->
    back cabin wall (Z=2.35), with protagonist torso/face projecting out inside the cabin (Z=1.45..1.65)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall_side = np.clip((np.abs(xx - 0.50 * W) - 180.0) / 180.0, 0.0, 1.0)
    wall_top = np.clip((220.0 - yy) / 220.0, 0.0, 1.0)
    concavity = np.maximum(wall_side, wall_top)
    z_room = 2.35 - 1.10 * concavity
    # Protagonist figure depth standing inside cabin
    bm = np.clip(bust_mask[:, :, 0], 0.0, 1.0)
    face_bump = np.exp(-(((xx - 0.50 * W) / 130.0) ** 2 + ((yy - 0.42 * H) / 170.0) ** 2))
    z_prot = 1.68 - 0.22 * face_bump
    z_cabin = z_room * (1.0 - bm) + z_prot * bm
    return cv2.GaussianBlur(z_cabin.astype(np.float32), (21, 21), 5.0)


def _render_protagonist_pose_3d(canvas, z_cabin, style_key, i_frame, hatch_map):
    """Render one 12-fps animated 3D perspective cabin frame around the Cubist protagonist."""
    t_tail = i_frame / 24.0
    # Smoothly dolly slightly forward as we step through the archway + breathe/settle into cabin
    dolly_enter = float(R.smoothstep(8.0, 115.0, i_frame))
    dolly_hold = float(R.smoothstep(115.0, 515.0, i_frame))
    tz = -0.14 * (1.0 - dolly_enter) + 0.14 * dolly_hold
    tx = 0.016 * np.sin(2.0 * np.pi * t_tail / 7.5)
    ty = 0.010 * np.cos(2.0 * np.pi * t_tail / 6.0)
    yaw = 1.35 * np.sin(2.0 * np.pi * t_tail / 7.5)
    pitch = 0.65 * np.cos(2.0 * np.pi * t_tail / 6.0)

    out_u8, _ = _project_3d_layer(
        np.clip(canvas["base_bgr"], 0, 255).astype(np.uint8),
        z_cabin,
        (0.50 * W, 0.40 * H),
        tx=tx, ty=ty, tz=tz, pitch=pitch, yaw=yaw, roll=0.0,
    )
    out = out_u8.astype(np.float32)

    # Story-specific 12-fps atmospheric cabin lighting
    if style_key == "nansen":
        wave = 0.5 + 0.5 * np.sin(0.012 * canvas["xx"] - 0.008 * canvas["yy"] + 2.0 * np.pi * t_tail / 4.5)
        aur = (canvas["ceil_w"] * (0.45 + 0.55 * wave))[:, :, None]
        out += aur * np.array([28.0, 34.0, 10.0], dtype=np.float32) * 0.65
    elif style_key == "shackleton":
        flicker = 0.65 + 0.35 * np.sin(2.0 * np.pi * t_tail / 2.9 + 0.7 * np.sin(2.0 * np.pi * t_tail / 1.3))
        lantern = (canvas["ceil_w"] * flicker)[:, :, None]
        out += lantern * np.array([10.0, 24.0, 38.0], dtype=np.float32) * 0.60
    else:
        warm_mix = float(R.smoothstep(3.0, 18.0, t_tail))
        cool_pulse = (1.0 - warm_mix) * canvas["ceil_w"][:, :, None] * np.array([26.0, 24.0, 6.0], dtype=np.float32) * 0.45
        warm_glow = warm_mix * (0.35 + 0.65 * canvas["ceil_w"])[:, :, None] * np.array([-8.0, 10.0, 24.0], dtype=np.float32) * 0.60
        out = out + cool_pulse + warm_glow

    out *= canvas["vignette"]
    out *= (1.0 + (0.045 * canvas["w_hatch"] * hatch_map)[:, :, None])
    out = out * (1.0 - canvas["ink_contour"]) + np.array([14.0, 16.0, 18.0], dtype=np.float32) * canvas["ink_contour"]
    return out


def _compose_tail_door_frame(styled_cabin_f32, door_plate, l3, style_key, i_frame):
    """Open ONLY the inner recessed doors inside Archway Bounds while 3D camera dollies through the archway."""
    styled = styled_cabin_f32.copy()
    if i_frame < 105:
        x_l, x_r, y_t, y_b = l3["arch_bounds"]
        cx = l3["seam_x"]
        prog = float(R.smoothstep(10.0, 86.0, i_frame))
        dolly = float(R.smoothstep(12.0, 102.0, i_frame))

        # Assemble composite Door + Outer Archway buffer at pre-projection plate before 3D dolly!
        plate = np.zeros((H, W, 3), dtype=np.float32)
        plate_a = np.zeros((H, W), dtype=np.float32)

        # 1. Inner recessed double doors clipped cleanly to portal interior [x_l..x_r, y_t..y_b]
        lw = cx - x_l
        rw = x_r - cx
        shift_l = int(round(prog * (lw + 6)))
        shift_r = int(round(prog * (rw + 6)))

        left_open_x = max(x_l, cx - shift_l)
        right_open_x = min(x_r, cx + shift_r)

        if left_open_x > x_l:
            vis_lw = left_open_x - x_l
            plate[y_t:y_b, x_l:left_open_x] = door_plate[y_t:y_b, x_l + shift_l : x_l + shift_l + vis_lw]
            plate_a[y_t:y_b, x_l:left_open_x] = 1.0
            # Dark 3D door leaf thickness bevel along inner receding edge
            bx0 = max(x_l, left_open_x - 7)
            plate[y_t:y_b, bx0:left_open_x] *= 0.45

        if right_open_x < x_r:
            vis_rw = x_r - right_open_x
            plate[y_t:y_b, right_open_x:x_r] = door_plate[y_t:y_b, cx : cx + vis_rw]
            plate_a[y_t:y_b, right_open_x:x_r] = 1.0
            bx1 = min(x_r, right_open_x + 7)
            plate[y_t:y_b, right_open_x:bx1] *= 0.45

        # Center closed-seam trim for initial hold frames i <= 10
        if i_frame <= 10:
            plate[y_t:y_b, cx - 2 : cx + 2] = 12.0

        # 2. Outer architectural portal archway overlay (strictly intact stone/wood/steel surround + barometer/lantern/telemetry)
        aa = l3["arch_alpha"]
        plate = plate * (1.0 - aa[:, :, None]) + door_plate.astype(np.float32) * aa[:, :, None]
        plate_a = np.maximum(plate_a, aa)

        # 3. 3D camera dolly through archway (Tz: 0.0 -> 0.76) with subtle walk step
        tz = 0.76 * dolly
        t_tail = i_frame / 24.0
        tx, ty, pitch, yaw, roll = _walk_kinematics(t_tail, step_hz=1.1, amp=0.35 * dolly)
        w_plate, w_alpha = _project_3d_layer(
            np.clip(plate, 0, 255).astype(np.uint8),
            l3["z_map"],
            l3["vp"],
            tx=tx, ty=ty, tz=tz, pitch=pitch, yaw=yaw, roll=roll,
            alpha=plate_a, border_const=True,
        )

        # Subtle interior door-frame cast shadow onto protagonist cabin behind
        sh_a = np.clip(cv2.GaussianBlur(w_alpha, (31, 31), 9.0) * 0.55, 0.0, 0.55)[:, :, None]
        styled *= (1.0 - sh_a)
        styled = styled * (1.0 - w_alpha[:, :, None]) + w_plate.astype(np.float32) * w_alpha[:, :, None]

    return styled


def _render_themed_tail(vcfg, l3, shader):
    """Render 516-frame 3D archway walk-through -> 3D concave protagonist cabin + Outro card."""
    key = vcfg["key"]
    style = vcfg["style"]
    out_tail = os.path.join(TRI_DIR, f"tail_{key}_themed.mp4")
    outro_bgr, outro_a = BTC.render_story_card(vcfg["outro"], vcfg["accent"], is_intro=False)

    hatch_bank = R.build_hatching_bank(12, H, W, seed=99)
    canvas = BTC._prepare_protagonist_canvas(style)
    z_cabin = _build_cabin_depth(canvas["base_bgr"], canvas["bust_mask"])
    door_plate = _build_cubist_door_plate(l3, style, vcfg["head_dur"], shader, hatch_bank[0]).astype(np.float32)

    n_frames = 516
    rng = np.random.default_rng(91 + sum(ord(c) for c in key))

    enc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "14", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-r", "24", out_tail, "-y",
        ],
        stdin=subprocess.PIPE,
    )

    cached_prot = None
    for i in range(n_frames):
        pose_idx = i // 2
        hatch = hatch_bank[pose_idx % len(hatch_bank)]
        if i % 2 == 0 or cached_prot is None:
            cached_prot = _render_protagonist_pose_3d(canvas, z_cabin, style, i, hatch)

        styled = _compose_tail_door_frame(cached_prot, door_plate, l3, style, i)

        t_tail = i / 24.0
        ca = BTC.card_alpha_at(t_tail, 4.2, 5.6, 17.5, 19.2)
        if ca > 0.001:
            a_eff = outro_a * ca
            styled = styled * (1.0 - a_eff) + outro_bgr * a_eff

        styled += rng.normal(0.0, 1.6, (H, W, 1)).astype(np.float32)
        enc.stdin.write(np.clip(styled, 0, 255).astype(np.uint8).tobytes())

    enc.stdin.close()
    enc.wait()
    print(f"3D Themed Cubist tail ({style}): {n_frames} frames -> {out_tail}")
    return out_tail


def run_preview():
    """Render quick layer & 3D motion progression diagnostics before running full encode."""
    os.makedirs(DIAG_DIR, exist_ok=True)
    shader = BTF.TrilogyShaderEngine()
    snow_bank = _build_snow_bank(16)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hatch_bank = R.build_hatching_bank(12, H, W, seed=99)

    layer_rows = []
    strip_rows = []
    tw, th = 225, 300

    for vcfg in BTF.VIDEOS:
        k = vcfg["key"]
        style = vcfg["style"]
        layers = _load_arrival_layers(k)
        l1, l2, l3 = layers

        # 1. Clean layer audit: Arr1 portal only | Arr2 clean bg | Arr3 hand matte | Arr3 clean door
        a1_vis = np.clip(l1["portal_bgr"].astype(np.float32) * l1["portal_alpha"][:, :, None] + 28 * (1 - l1["portal_alpha"][:, :, None]), 0, 255).astype(np.uint8)
        h3_vis = np.clip(l3["hand_bgr"].astype(np.float32) * l3["hand_alpha"][:, :, None] + 28 * (1 - l3["hand_alpha"][:, :, None]), 0, 255).astype(np.uint8)
        layer_rows.append(np.hstack([
            cv2.resize(a1_vis, (240, 320)),
            cv2.resize(l2["bg_bgr"], (240, 320)),
            cv2.resize(h3_vis, (240, 320)),
            cv2.resize(l3["clean_bgr"], (240, 320)),
        ]))

        # 2. 8-frame 3D motion progression strip across Arr1 -> Arr2 -> Arr3 hand reach -> Archway walk-through -> Cabin
        canvas = BTC._prepare_protagonist_canvas(style)
        z_cabin = _build_cabin_depth(canvas["base_bgr"], canvas["bust_mask"])
        door_plate = _build_cubist_door_plate(l3, style, vcfg["head_dur"], shader, hatch_bank[0]).astype(np.float32)
        outro_bgr, outro_a = BTC.render_story_card(vcfg["outro"], vcfg["accent"], is_intro=False)

        frames_8 = []
        head_beats = [
            ("1.Arr1 open", 0.08),
            ("2.Arr1 exit", 0.27),
            ("3.Arr2 walk", 0.45),
            ("4.Arr2->3 zoom", 0.62),
            ("5.Arr3 hand", 0.87),
            ("6.Arr3 clear", 0.99),
        ]
        for lbl, u in head_beats:
            f32 = _render_arrival_frame_3d(layers, style, u, vcfg["head_dur"] - 8.0 + u * 8.0, int(u * 60), shader, snow_bank, xx, yy)
            sm = cv2.resize(np.clip(f32, 0, 255).astype(np.uint8), (tw, th))
            cv2.putText(sm, f"{k}:{lbl}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1)
            frames_8.append(sm)

        for lbl, t_idx in [("7.Arch dolly", 56), ("8.3D Cabin+Out", 260)]:
            prot = _render_protagonist_pose_3d(canvas, z_cabin, style, t_idx, hatch_bank[(t_idx // 2) % 12])
            comp = _compose_tail_door_frame(prot, door_plate, l3, style, t_idx)
            ca = BTC.card_alpha_at(t_idx / 24.0, 4.2, 5.6, 17.5, 19.2)
            if ca > 0.001:
                comp = comp * (1.0 - outro_a * ca) + outro_bgr * (outro_a * ca)
            sm = cv2.resize(np.clip(comp, 0, 255).astype(np.uint8), (tw, th))
            cv2.putText(sm, f"{k}:{lbl}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1)
            frames_8.append(sm)

        strip_rows.append(np.hstack(frames_8))

    audit_p = os.path.join(DIAG_DIR, "themed_ending_3d_layer_audit.jpg")
    strip_p = os.path.join(DIAG_DIR, "themed_ending_3d_motion_strip.jpg")
    cv2.imwrite(audit_p, np.vstack(layer_rows))
    cv2.imwrite(strip_p, np.vstack(strip_rows))
    print("Wrote", audit_p)
    print("Wrote", strip_p)


def run_all():
    os.makedirs(TRI_DIR, exist_ok=True)
    os.makedirs(DIAG_DIR, exist_ok=True)
    with open(BTF.QUADS_PATH) as f:
        tracks = json.load(f)
    with open(BTF.FULL_QUADS_PATH) as f:
        full_data = json.load(f)

    shader = BTF.TrilogyShaderEngine()
    snow_bank = _build_snow_bank(16)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for vcfg in BTF.VIDEOS:
        k = vcfg["key"]
        sched = BTF.build_pose_schedule(vcfg["speed_mode"], vcfg["head_dur"], tracks, full_data, screen_hold=4)
        f_stop = next(i for i, (p, _, _, _) in enumerate(sched) if p >= CAR_STOP_POSE)
        layers = _load_arrival_layers(k)

        head_p = _render_themed_head(vcfg, sched, f_stop, shader, layers, snow_bank, xx, yy)
        tail_p = _render_themed_tail(vcfg, layers[2], shader)

        hd = vcfg["head_dur"]
        tot = vcfg["total_dur"]
        base_name = os.path.splitext(os.path.basename(vcfg["out"]))[0]
        themed_out = os.path.join(B.OUT, f"{base_name}_themed-ending.mp4")

        filt = (
            f"[0:v][1:v]xfade=transition=fade:duration=0.50:offset={hd - 0.50:.2f},"
            f"fade=t=in:st=0:d=0.9,fade=t=out:st={tot - 4.00:.2f}:d=4.00,format=yuv420p[v]"
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", head_p, "-i", tail_p, "-i", vcfg["audio"],
                "-filter_complex", filt, "-map", "[v]", "-map", "2:a",
                "-c:v", "libx264", "-crf", "15", "-preset", "fast",
                "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                "-t", f"{tot:.2f}", "-movflags", "+faststart", themed_out, "-y",
            ],
            check=True,
        )
        print(f"MUXED 3D THEMED FINAL: {themed_out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "preview":
        run_preview()
    else:
        run_all()
