#!/usr/bin/env python3
"""Render & mux all 3 trilogy films in a single shared bilateral/edge pass:
  01: Nansen     - Into the White Silence     (280.00s, fast hyper-lapse, ice-blue cyan lithograph)
  02: Shackleton - Endurance in the Dark      (240.00s, heavy slow/medium celluloid noir + amber halation)
  03: Mars500    - 520 Days to Another World  (255.00s, slow->fast->slow ramp, electric-teal CAD blueprint -> gold)

Usage:
  uv run --with opencv-python-headless --with numpy --with scipy --with pillow python -m pipeline.build_trilogy_films preview
  uv run --with opencv-python-headless --with numpy --with scipy --with pillow python -m pipeline.build_trilogy_films all
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import build_trilogy_tails as BTC, fetch_archival_media as FEA, media_base as B, night_chalk_shader as R, paths, planar_tracker_segments as TP, quad_overlay as T, track_full_drive_display as TFD

W, H = R.W, R.H
N_SRC_POSES = 3198
TAIL_PATH = os.path.join(B.WORK, "final3_parts", "tail_cubist_chalk.mp4")
QUADS_PATH = paths.QUADS_EXPEDITION
FULL_QUADS_PATH = TFD.OUT_JSON_PATH
SCR_DIR = os.path.join(B.WORK, "trilogy_assets", "screens")
TRI_DIR = os.path.join(B.WORK, "trilogy_assets", "films")
DIAG_DIR = os.path.join(B.WORK, "track_diag")

STAGE_STARTS = {stg: s0 for stg, s0, _ in TFD.STAGE_BOUNDS}

HEAD_SIGNS = [
    (1, 11.90, 12.60),
    (2, 91.15, 91.85),
    (3, 141.40, 141.85),
    (4, 204.20, 204.60),
    (5, 237.50, 238.15),
    (6, 240.65, 241.35),
]

VIDEOS = [
    {
        "key": "nansen",
        "total_dur": 280.00,
        "head_dur": 259.00,  # + 21.50 tail - 0.50 xfade = 280.00
        "audio": os.path.join(B.WORK, "audio288", "trilogy_01_nansen_master.wav"),
        "out": os.path.join(B.OUT, "01_nansen_into-the-white-silence.mp4"),
        "speed_mode": "nansen",
        "style": "nansen",
        "accent": (125, 232, 255),
        "intro": (
            "NORWEGIAN POLAR EXPEDITION • 1893–1896",
            "INTO THE WHITE SILENCE",
            "Fridtjof Nansen • Farthest North (1897)",
            "In 1893, Fridtjof Nansen deliberately froze his ship Fram into the Arctic pack ice to drift toward the North Pole. When the drift passed too far south, he and Hjalmar Johansen left the ship with kayaks and dog sleds into uncharted ice. You will hear Nansen's own expedition journal from Farthest North, recounting their solitary march to 86°13.6' N.",
        ),
        "outro": (
            "MAN WANTS TO KNOW",
            "Fridtjof Nansen — Farthest North (1897) • 86°13.6' N",
            "Nansen burned his bridges not for conquest, but out of pure curiosity—stepping off a warm ship into endless white silence with one companion. His drift proves that when every certainty vanishes, the quiet refusal to stop questioning is what defines the human spirit.",
        ),
    },
    {
        "key": "shackleton",
        "total_dur": 240.00,
        "head_dur": 219.00,  # + 21.50 tail - 0.50 xfade = 240.00
        "audio": os.path.join(B.WORK, "audio288", "trilogy_02_shackleton_master.wav"),
        "out": os.path.join(B.OUT, "02_shackleton_endurance-in-the-dark.mp4"),
        "speed_mode": "shackleton",
        "style": "shackleton",
        "accent": (255, 196, 98),
        "intro": (
            "IMPERIAL TRANS-ANTARCTIC EXPEDITION • 1914–1916",
            "ENDURANCE IN THE DARK",
            "Sir Ernest Shackleton • South! (1919) & 1910 Phonograph",
            "In January 1915, Ernest Shackleton's ship Endurance was trapped and crushed by millions of tons of Weddell Sea pack ice, stranding 28 men on drifting floes in the Antarctic dark. You will hear excerpts from Shackleton's memoir South! alongside his actual 1910 Edison phonograph voice—speaking across a century about survival and bringing every man home.",
        ),
        "outro": (
            "NOT A LIFE LOST",
            "Sir Ernest Shackleton — South! (1919) • Weddell Sea & South Georgia",
            "Shackleton failed his original goal before he ever touched the continent, yet forged history's greatest triumph of leadership. When the ice crushed his ship in the dark, he traded ambition for a harder promise: to fight despair day by day and bring all twenty-eight men home alive.",
        ),
    },
    {
        "key": "mars500",
        "total_dur": 255.00,
        "head_dur": 234.00,  # + 21.50 tail - 0.50 xfade = 255.00
        "audio": os.path.join(B.WORK, "audio288", "trilogy_03_mars500_master.wav"),
        "out": os.path.join(B.OUT, "03_mars500_520-days-to-another-world.mp4"),
        "speed_mode": "mars500",
        "style": "mars500",
        "accent": (95, 245, 235),
        "intro": (
            "ESA MARS500 ISOLATION MISSION • 2010–2011",
            "520 DAYS TO ANOTHER WORLD",
            "Crewmembers Diego Urbina & Romain Charles • Moscow IBMP",
            "Between June 2010 and November 2011, six volunteers sealed themselves inside a steel habitat in Moscow for 520 days—simulating a full mission to Mars without sunlight or real-time contact with Earth. You will hear the actual recorded voices of ESA crewmembers Diego Urbina and Romain Charles reflecting from inside the capsule before the hatch opened.",
        ),
        "outro": (
            "READY FOR THE DAY WE GO",
            "ESA Mars500 Mission — 520 Days Sealed • Moscow IBMP (2010–2011)",
            "Stripped of storms and frostbite, Mars500 tested the frontier we will face next: five hundred twenty days of steel walls, twenty-minute radio delays, and no sun. Six strangers proved that patience, humor, and brotherhood can carry humanity across the void to another planet.",
        ),
    },
]


def _pingpong_scr_idx(pose_idx, stage):
    off = max(0, int(pose_idx) - STAGE_STARTS.get(int(stage), 0))
    cyc = (off * 2) % 142
    return cyc if cyc < 72 else 142 - cyc


def _probe_nb_frames(path):
    if not os.path.isfile(path):
        return -1
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=nk=1:nw=1", path,
        ],
        text=True,
    ).strip()
    return int(out) if out.isdigit() else -1


def load_panel_poses(tracks):
    """Map pose_idx -> (panel, quad_np_f32) for every 12-fps pose inside P1..P6."""
    pose_panel = {}
    for p, st, en in HEAD_SIGNS:
        start_f = int(round(st * 24.0))
        quads = tracks[str(p)]["quads"]
        n_q = len(quads)
        end_f = start_f + n_q
        # Group source frames into unique pose_idx = f // 2
        for f_idx in range(start_f, end_f):
            p_idx = f_idx // 2
            q = quads[f_idx - start_f]
            if q is not None and p_idx not in pose_panel:
                pose_panel[p_idx] = (p, np.array(q, dtype=np.float32))
    return pose_panel


def build_pose_schedule(mode, head_dur, tracks, full_data, screen_hold=4):
    """Build monotonic schedule list of length int(round(head_dur*24)):
      entries: (pose_idx, stage_panel_or_0, screen_local_idx_0_to_71, quad_or_None)
    Keep exact P1..P6 slow-mo timing while showing active Stage 1..6 across ALL visible screen poses.
    """
    n_out = int(round(head_dur * 24.0))
    pose_panel = load_panel_poses(tracks)
    full_quads = [np.array(q, dtype=np.float32) if q is not None else None for q in full_data["quads"]]
    full_stages = full_data["stages"]

    def _entry_for_pose(p_idx):
        fq = full_quads[p_idx]
        stg = int(full_stages[p_idx])
        if fq is not None and stg > 0:
            return (p_idx, stg, _pingpong_scr_idx(p_idx, stg), fq)
        return (p_idx, 0, 0, None)

    # Identify contiguous blocks: screen blocks (each pose held `screen_hold` frames) and driving blocks
    blocks = []
    cur_kind = 0 if 0 not in pose_panel else pose_panel[0][0]
    b_start = 0
    for p_idx in range(1, N_SRC_POSES):
        k = 0 if p_idx not in pose_panel else pose_panel[p_idx][0]
        if k != cur_kind:
            blocks.append((cur_kind, b_start, p_idx))
            cur_kind = k
            b_start = p_idx
    blocks.append((cur_kind, b_start, N_SRC_POSES))

    screen_frames_total = sum((e - s) * screen_hold for k, s, e in blocks if k != 0)
    drive_budget = n_out - screen_frames_total

    # Compute continuous speed multiplier per source driving pose to weight time allocation
    def drive_rate(u):
        # u in [0, 1] across source drive
        if mode == "nansen":
            # Crisp relentless hyper-lapse that eases gently for landing/return
            return 1.08 + 0.16 * np.sin(np.pi * u) - 0.30 * float(R.smoothstep(0.75, 0.98, u))
        if mode == "shackleton":
            # Deliberate, heavier slow/medium pace on traffic/gate/arrival, faster across open cruise
            return 1.35 - 0.52 * float(R.smoothstep(0.32, 0.45, u)) - 0.45 * float(R.smoothstep(0.76, 0.98, u))
        if mode == "mars500":
            # Distinct variable ramp: 0.62x slow dusk launch -> 1.65x hyperlapse isolation -> 0.58x slow hatch return
            mid = float(R.smoothstep(0.14, 0.30, u)) * (1.0 - float(R.smoothstep(0.68, 0.84, u)))
            return 0.60 + 1.05 * mid
        return 1.0

    # Inverse rate = dwell density -> total dwell per driving pose
    raw_dwell = np.zeros(N_SRC_POSES, dtype=np.float64)
    for k, s, e in blocks:
        if k == 0:
            for p_idx in range(s, e):
                raw_dwell[p_idx] = 1.0 / max(0.25, drive_rate(p_idx / (N_SRC_POSES - 1)))

    total_dwell = float(raw_dwell.sum())
    scaled_dwell = raw_dwell * (drive_budget / total_dwell)

    # Allocate exact integer frame counts to each driving block summing strictly to drive_budget
    exact_counts = [float(scaled_dwell[s:e].sum()) for k, s, e in blocks if k == 0]
    int_counts = [int(np.floor(c)) for c in exact_counts]
    shortfall = drive_budget - sum(int_counts)
    frac_order = np.argsort([c - ic for c, ic in zip(exact_counts, int_counts)])[::-1]
    for j in range(shortfall):
        int_counts[int(frac_order[j])] += 1

    block_alloc = {}
    di = 0
    for idx, (k, s, e) in enumerate(blocks):
        if k != 0:
            block_alloc[idx] = (e - s) * screen_hold
        else:
            block_alloc[idx] = int_counts[di]
            di += 1

    sched = []
    for idx, (k, s, e) in enumerate(blocks):
        alloc = block_alloc[idx]
        n_p = e - s
        if k != 0:
            # 12-fps locked slow motion across signature beat + persistent stage portal
            for p_offset in range(n_p):
                p_idx = s + p_offset
                _, q = pose_panel[p_idx]
                scr_idx = _pingpong_scr_idx(p_idx, k)
                for _ in range(screen_hold):
                    sched.append((p_idx, k, scr_idx, q))
        else:
            # Allocate frames monotonically along pose sequence s..e-1 using CDF of dwell
            w = scaled_dwell[s:e]
            cdf = np.cumsum(w)
            cdf = cdf / (cdf[-1] + 1e-12)
            for out_j in range(alloc):
                u = (out_j + 0.5) / max(1, alloc)
                # Lock consecutive 2-frame groups when speed <= 1 to preserve crisp 12-fps look
                if out_j % 2 == 1 and alloc >= n_p * 1.4 and len(sched) > 0:
                    prev_p = sched[-1][0]
                    if s <= prev_p < e:
                        sched.append(_entry_for_pose(prev_p))
                        continue
                rel = int(np.searchsorted(cdf, u, side="right"))
                rel = max(0, min(n_p - 1, rel))
                sched.append(_entry_for_pose(s + rel))

    assert len(sched) == n_out, f"{mode}: {len(sched)} != {n_out}"
    return sched


def render_overlay_card(card_tuple, accent_rgb, is_intro=True):
    return BTC.render_story_card(card_tuple, accent_rgb, is_intro=is_intro)


def card_alpha_at(t, t_start, fade_in, hold_end, fade_out):
    return BTC.card_alpha_at(t, t_start, fade_in, hold_end, fade_out)


class TrilogyShaderEngine:
    """Computes shared bilateral smoothing + CIE Lab + edges once per pose, then branches per video style."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        self.hatch_bank = R.build_hatching_bank(12, H, W)
        self.xp = np.array([0.0, 14.0, 22.0, 42.0, 85.0, 145.0, 210.0, 255.0], dtype=np.float32)
        self.fp = np.array([0.0,  4.0,  9.0, 44.0, 106.0, 184.0, 244.0, 255.0], dtype=np.float32)

        # Vignette map for Shackleton celluloid look
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rx = (xx - W * 0.5) / (W * 0.55)
        ry = (yy - H * 0.5) / (H * 0.55)
        self.vignette = np.clip(1.0 - 0.24 * (rx * rx + ry * ry), 0.72, 1.0)[:, :, None]

    def prepare_base(self, raw_bgr):
        sm = cv2.bilateralFilter(raw_bgr, 9, 45, 10)
        sm = cv2.bilateralFilter(sm, 7, 38, 8)
        sm = cv2.bilateralFilter(sm, 7, 32, 8)

        lab = cv2.cvtColor(sm, cv2.COLOR_BGR2Lab)
        L_u8, a_u8, b_u8 = cv2.split(lab)
        L = L_u8.astype(np.float32)
        L_clahe = self.clahe.apply(L_u8).astype(np.float32)
        L_base = 0.70 * L + 0.30 * L_clahe
        L_curve = np.interp(L_base, self.xp, self.fp).astype(np.float32)

        g1 = cv2.GaussianBlur(L_base, (0, 0), 0.75)
        g2 = cv2.GaussianBlur(L_base, (0, 0), 2.10)
        dog = g1 - g2
        sx = cv2.Sobel(L_base, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(L_base, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sx, sy) * 0.25

        sky_gate = R.smoothstep(15.0, 27.0, L)
        cad_gate = R.smoothstep(10.0, 22.0, L)
        ink_raw = np.clip((grad - 13.0) / 42.0, 0.0, 1.0) * 0.85 + np.clip((-dog - 1.6) / 6.5, 0.0, 1.0) * 0.72
        ink_bold = 0.65 * ink_raw + 0.35 * cv2.dilate(ink_raw, np.ones((2, 2), np.uint8))
        ink = np.clip(ink_bold * sky_gate, 0.0, 0.92)
        ink_cad = np.clip(ink_bold * cad_gate, 0.0, 0.95)

        return dict(
            L=L, L_base=L_base, L_curve=L_curve,
            a_off=a_u8.astype(np.float32) - 128.0,
            b_off=b_u8.astype(np.float32) - 128.0,
            dog=dog, ink=ink, ink_cad=ink_cad,
            night_gate=R.smoothstep(14.0, 28.0, L),
        )

    def render_style(self, base, pose_idx, style, t_sec=0.0):
        L = base["L"]
        L_curve = base["L_curve"]
        a_off, b_off = base["a_off"], base["b_off"]
        night_gate = base["night_gate"]
        C = np.hypot(a_off, b_off)
        neutral_mask = 1.0 - 0.50 * np.clip((C - 28.0) / 38.0, 0.0, 1.0)
        hatch = self.hatch_bank[pose_idx % len(self.hatch_bank)]

        if style == "nansen":
            # Distinct Arctic Silver-Ice, Alabaster Chalk & Aurora Borealis palette
            step = 18.0
            q_idx = np.floor(L_curve / step)
            phase = (L_curve / step) - q_idx - 0.5
            L_q = (q_idx + 0.5 + 0.5 * np.tanh(phase * 4.8) / np.tanh(2.4)) * step
            L_toned = np.clip(L_curve * (1.0 - 0.66 * night_gate) + L_q * (0.66 * night_gate), 0.0, 255.0)
            L_ice = np.clip(L_toned * 1.08 + 6.0 * night_gate, 0.0, 255.0)

            b_mapped = np.where(b_off > 0.0, -0.58 * b_off, 0.85 * b_off)
            a_mapped = np.where(a_off > 0.0, -0.35 * a_off, 0.75 * a_off)
            ice_w = R.smoothstep(16.0, 48.0, L_ice) * (1.0 - 0.65 * R.smoothstep(185.0, 245.0, L_ice))
            a_new = np.clip(128.0 + a_mapped * 1.15 - 9.5 * ice_w, 0.0, 255.0)
            b_new = np.clip(128.0 + b_mapped * 1.25 - 26.0 * ice_w - 8.0 * night_gate, 0.0, 255.0)

            out = cv2.cvtColor(cv2.merge([L_ice.astype(np.uint8), a_new.astype(np.uint8), b_new.astype(np.uint8)]), cv2.COLOR_Lab2BGR).astype(np.float32)
            w_hatch = R.smoothstep(18.0, 42.0, L_ice) * (1.0 - R.smoothstep(145.0, 215.0, L_ice))
            out *= (1.0 + (0.130 * w_hatch * hatch)[:, :, None])
            out *= (1.0 - (base["ink"] * 0.96)[:, :, None])

            # Aurora Borealis cyan-emerald halation on bright lights + icy chalk rim highlights
            bright = np.clip((L_ice - 185.0) / 70.0, 0.0, 1.0)
            aurora_glow = cv2.GaussianBlur(bright, (0, 0), 8.0)
            out += aurora_glow[:, :, None] * np.array([235.0, 250.0, 115.0], dtype=np.float32) * 0.42
            rim = np.clip((base["dog"] - 2.2) / 9.0, 0.0, 1.0) * R.smoothstep(44.0, 98.0, L_ice)
            out += rim[:, :, None] * np.array([255.0, 248.0, 185.0], dtype=np.float32) * 0.48
            return np.clip(out, 0.0, 255.0)

        if style == "shackleton":
            # Heavy 1915 sepia/amber celluloid noir + headlight halation bloom + subtle vignette
            step = 32.0
            q_idx = np.floor(L_curve / step)
            phase = (L_curve / step) - q_idx - 0.5
            L_q = (q_idx + 0.5 + 0.5 * np.tanh(phase * 3.8) / np.tanh(1.9)) * step
            L_toned = np.clip(L_curve * (1.0 - 0.60 * night_gate) + L_q * (0.60 * night_gate), 0.0, 255.0)
            sat_boost = 1.95 * R.smoothstep(15.0, 34.0, L)
            warm_w = R.smoothstep(24.0, 68.0, L_toned) * neutral_mask
            cool_w = R.smoothstep(15.0, 28.0, L) * (1.0 - R.smoothstep(24.0, 54.0, L_toned)) * neutral_mask
            a_new = np.clip(128.0 + a_off * sat_boost + 12.5 * warm_w + 2.0 * cool_w, 0.0, 255.0)
            b_new = np.clip(128.0 + b_off * sat_boost + 26.0 * warm_w + 4.0 * cool_w, 0.0, 255.0)

            out = cv2.cvtColor(cv2.merge([L_toned.astype(np.uint8), a_new.astype(np.uint8), b_new.astype(np.uint8)]), cv2.COLOR_Lab2BGR).astype(np.float32)
            w_hatch = R.smoothstep(20.0, 42.0, L_toned) * (1.0 - R.smoothstep(115.0, 175.0, L_toned))
            out *= (1.0 + (0.080 * w_hatch * hatch)[:, :, None])
            out *= (1.0 - base["ink"][:, :, None])

            # Warm amber halation glow on bright lights & wet asphalt
            bright = np.clip((L_toned - 180.0) / 75.0, 0.0, 1.0)
            glow = cv2.GaussianBlur(bright, (0, 0), 9.0)
            out += glow[:, :, None] * np.array([75.0, 165.0, 255.0], dtype=np.float32) * 0.52
            out *= self.vignette
            return np.clip(out, 0.0, 255.0)

        # style == "mars500": Midnight indigo blueprint with glowing electric-teal vector lines,
        # gradually thawing to luminous gold over the final 35s (hatch opening: t_sec 195..234s)
        warm_mix = float(R.smoothstep(195.0, 232.0, t_sec))
        step = 26.0
        q_idx = np.floor(L_curve / step)
        phase = (L_curve / step) - q_idx - 0.5
        L_q = (q_idx + 0.5 + 0.5 * np.tanh(phase * 4.2) / np.tanh(2.1)) * step
        L_toned = np.clip(L_curve * (1.0 - 0.58 * night_gate) + L_q * (0.58 * night_gate), 0.0, 255.0)
        # Deepen dark indigo blueprint midtones slightly in isolation phase
        L_adj = L_toned * (0.86 + 0.14 * warm_mix)

        sat_boost = 1.90 * R.smoothstep(15.0, 34.0, L)
        warm_w = R.smoothstep(26.0, 72.0, L_adj) * neutral_mask
        cool_w = R.smoothstep(15.0, 28.0, L) * (1.0 - R.smoothstep(26.0, 56.0, L_adj)) * neutral_mask
        a_w = -15.0 * (1.0 - warm_mix) + 8.0 * warm_mix
        b_w = -12.0 * (1.0 - warm_mix) + 22.0 * warm_mix
        a_c = 4.0 * (1.0 - warm_mix) - 2.0 * warm_mix
        b_c = -22.0 * (1.0 - warm_mix) - 4.0 * warm_mix
        a_new = np.clip(128.0 + a_off * sat_boost + a_w * warm_w + a_c * cool_w, 0.0, 255.0)
        b_new = np.clip(128.0 + b_off * sat_boost + b_w * warm_w + b_c * cool_w, 0.0, 255.0)

        out = cv2.cvtColor(cv2.merge([L_adj.astype(np.uint8), a_new.astype(np.uint8), b_new.astype(np.uint8)]), cv2.COLOR_Lab2BGR).astype(np.float32)
        w_hatch = R.smoothstep(20.0, 40.0, L_adj) * (1.0 - R.smoothstep(115.0, 175.0, L_adj))
        out *= (1.0 + (0.065 * w_hatch * hatch)[:, :, None])

        # Vector outline: glowing electric-teal blueprint strokes in dark phase -> charcoal ink in sunlit return
        ink = base["ink"]
        ink_cad = base["ink_cad"]
        teal_line = np.array([255.0, 245.0, 105.0], dtype=np.float32)
        out = out * (1.0 - (ink * warm_mix)[:, :, None])
        glow_edge = ink_cad * (1.0 - warm_mix)
        out = out * (1.0 - 0.60 * glow_edge[:, :, None]) + glow_edge[:, :, None] * teal_line * 1.12
        return np.clip(out, 0.0, 255.0)


def load_screens(story_key):
    """Load 6 72-frame animated screens (bezel-masked) for a story."""
    ink_bgr = np.array([14.0, 16.0, 20.0], dtype=np.float32)
    screens = {}
    for p in range(1, 7):
        mp4_p = os.path.join(SCR_DIR, f"{story_key}_screen{p}.mp4")
        cap = cv2.VideoCapture(mp4_p)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        fh, fw = frames[0].shape[:2]
        r = max(6, int(round(fh * 58.0 / 450.0)))
        outer_m = T._rounded_mask(fw, fh, r, pad=0)
        inner_m = T._rounded_mask(fw, fh, max(4, r - 12), pad=12)
        border_a = np.clip((outer_m.astype(np.float32) - inner_m.astype(np.float32)) / 255.0, 0.0, 1.0)[:, :, None]
        prepped = [
            np.clip(fr.astype(np.float32) * (1.0 - border_a) + ink_bgr * border_a, 0, 255).astype(np.uint8)
            for fr in frames
        ]
        screens[p] = (prepped, outer_m, inner_m)
    return screens


def composite_trilogy_screen(frame_f32, panel, scr_idx, q, screens, hatch_map, rng):
    prepped, outer_m, inner_m = screens[panel]
    face = prepped[scr_idx]
    fh, fw = face.shape[:2]
    src = np.array([(0, 0), (fw - 1, 0), (fw - 1, fh - 1), (0, fh - 1)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, q)

    warped = cv2.warpPerspective(face, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped = cv2.GaussianBlur(warped, (0, 0), 0.55).astype(np.float32)
    m = cv2.GaussianBlur(cv2.warpPerspective(outer_m, M, (W, H), flags=cv2.INTER_LINEAR), (0, 0), 0.65).astype(np.float32) / 255.0
    m_in = cv2.warpPerspective(inner_m, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    sel = m_in > 0.5
    if not sel.any():
        sel = m > 0.5
    if not sel.any():
        return frame_f32

    # Bright luminous LCD glow so headline & stage progress bar remain clearly intelligible
    env = cv2.GaussianBlur(cv2.cvtColor(np.clip(frame_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (31, 31), 10)
    lum_sign = float(env[sel].mean())
    lum_face = float(warped[sel].mean()) + 1e-6
    base_gain = min(max((lum_sign + 38.0) / lum_face, 0.78), 0.96)
    spatial_mod = np.clip(0.88 + 0.12 * (env / (lum_sign + 1e-6)), 0.78, 1.12)[:, :, None]

    wf = warped * base_gain * spatial_mod
    wf *= (1.0 + 0.032 * hatch_map[:, :, None])
    wf += rng.normal(0.0, 2.2, (H, W, 1)).astype(np.float32)
    a = (m * 0.98)[:, :, None]
    return frame_f32 * (1.0 - a) + np.clip(wf, 0.0, 255.0) * a


def restyle_cubist_tail(vcfg, out_tail):
    """Build narrative-aligned Cubist protagonist elevator ending + Outro story card."""
    BTC.render_single_tail(vcfg, out_tail)


def apply_intro_if_active(f32, t_out, intro_bgr, intro_a):
    ca = card_alpha_at(t_out, 0.6, 1.8, 9.6, 11.2)
    if ca > 0.001:
        a_eff = intro_a * ca
        return f32 * (1.0 - a_eff) + intro_bgr * a_eff
    return f32


def run_preview():
    os.makedirs(DIAG_DIR, exist_ok=True)
    with open(FULL_QUADS_PATH) as f:
        full_data = json.load(f)
    shader = TrilogyShaderEngine()
    rng = np.random.default_rng(29)
    hatch_tail = R.build_hatching_bank(12, H, W, seed=99)

    rows = []
    for vcfg in VIDEOS:
        key = vcfg["key"]
        screens = load_screens(key)
        intro_bgr, intro_a = render_overlay_card(vcfg["intro"], vcfg["accent"], is_intro=True)
        outro_bgr, outro_a = render_overlay_card(vcfg["outro"], vcfg["accent"], is_intro=False)

        # Sample: Intro (3.5s raw -> pose 42, Stage 1 persistent screen), Mid (96.2s raw -> pose 1154, Stage 2), P4 (pose 2453), Outro (Protagonist elevator @10.8s)
        tiles = []
        for lbl, t_raw, panel_override in [
            (f"{key}: Intro", 3.5, None),
            (f"{key}: S2 hold", 96.17, None),
            (f"{key}: P4", 204.40, 4),
        ]:
            raw_fr = R.grab_frame_at(R.HEAD_RAW, t_raw)
            pose_idx = int(round(t_raw * 24.0)) // 2
            base = shader.prepare_base(raw_fr)
            t_style = 210.0 if panel_override == 4 else (95.0 if "S2" in lbl else 4.0)
            f32 = shader.render_style(base, pose_idx, vcfg["style"], t_sec=t_style)
            stg = int(full_data["stages"][pose_idx])
            fq = full_data["quads"][pose_idx]
            if fq is not None and stg > 0:
                q = np.array(fq, dtype=np.float32)
                scr_idx = _pingpong_scr_idx(pose_idx, stg)
                f32 = composite_trilogy_screen(f32, stg, scr_idx, q, screens, shader.hatch_bank[pose_idx % 12], rng)
            if "Intro" in lbl:
                f32 = f32 * (1.0 - intro_a) + intro_bgr * intro_a
            u8 = np.clip(f32, 0, 255).astype(np.uint8)
            sm = cv2.resize(u8, (300, 400))
            cv2.putText(sm, lbl, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2)
            tiles.append(sm)

        # Sample Protagonist Cubist elevator tail + Outro card at i=260 (~10.8s)
        canvas = BTC._prepare_protagonist_canvas(vcfg["style"])
        tf = BTC.render_protagonist_pose(canvas, vcfg["style"], 260, hatch_tail[(260 // 2) % 12])
        tf = tf * (1.0 - outro_a) + outro_bgr * outro_a
        sm_t = cv2.resize(np.clip(tf, 0, 255).astype(np.uint8), (300, 400))
        cv2.putText(sm_t, f"{key}: Outro", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2)
        tiles.append(sm_t)
        rows.append(np.hstack(tiles))

    out_p = os.path.join(DIAG_DIR, "trilogy_preview_grid.jpg")
    cv2.imwrite(out_p, np.vstack(rows))
    print("Wrote", out_p)


def run_all():
    os.makedirs(TRI_DIR, exist_ok=True)
    os.makedirs(DIAG_DIR, exist_ok=True)
    with open(QUADS_PATH) as f:
        tracks = json.load(f)
    with open(FULL_QUADS_PATH) as f:
        full_data = json.load(f)

    shader = TrilogyShaderEngine()
    rng = np.random.default_rng(29)

    state = {}
    for vcfg in VIDEOS:
        k = vcfg["key"]
        sched = build_pose_schedule(vcfg["speed_mode"], vcfg["head_dur"], tracks, full_data, screen_hold=4)
        n_out = len(sched)
        head_out = os.path.join(TRI_DIR, f"head_{k}.mp4")
        head_prev = os.path.join(TRI_DIR, f"head_{k}_prev.mp4")

        # Nansen gets a full drive palette re-render; Shackleton & Mars500 reuse finalized drive frames > 11.5s
        force_full = (k == "nansen")
        prev_dec = None
        if not force_full:
            if os.path.isfile(head_prev) and _probe_nb_frames(head_prev) == n_out:
                prev_dec = subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-i", head_prev, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                    stdout=subprocess.PIPE,
                )
            elif os.path.isfile(head_out) and _probe_nb_frames(head_out) == n_out:
                shutil.move(head_out, head_prev)
                prev_dec = subprocess.Popen(
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
        intro_bgr, intro_a = render_overlay_card(vcfg["intro"], vcfg["accent"], is_intro=True)
        n_scr = sum(1 for _, p, _, _ in sched if p != 0)
        state[k] = {
            "cfg": vcfg,
            "sched": sched,
            "ptr": 0,
            "n_out": n_out,
            "enc": enc,
            "prev_dec": prev_dec,
            "head_prev": head_prev,
            "head_out": head_out,
            "screens": load_screens(k),
            "intro_bgr": intro_bgr,
            "intro_a": intro_a,
            "n_rendered": 0,
            "n_copied": 0,
        }
        mode_lbl = "FULL (new palette)" if force_full else ("SELECTIVE Intro<=11.5s + COPY" if prev_dec else "FULL")
        print(
            f"Scheduled {k}: {n_out} head frames ({vcfg['head_dur']:.2f}s), "
            f"screen_frames={n_scr}, mode={mode_lbl}"
        )

    # Single sequential decode of head_raw.mp4 reading every even pose (0..3197)
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", R.HEAD_RAW, "-vf", "fps=24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = W * H * 3
    for src_f in range(N_SRC_POSES * 2):
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        if src_f % 2 != 0:
            continue
        pose_idx = src_f // 2

        active = [st for st in state.values() if st["ptr"] < st["n_out"] and st["sched"][st["ptr"]][0] == pose_idx]
        if not active:
            continue

        # Check if any active video frame for pose_idx needs bilateral shader computation
        need_base = False
        for st in active:
            if st["prev_dec"] is None:
                need_base = True
                break
            j = st["ptr"]
            while j < st["n_out"] and st["sched"][j][0] == pose_idx:
                if (j / 24.0) <= 11.5:
                    need_base = True
                    break
                j += 1
            if need_base:
                break

        base = None
        if need_base:
            raw_fr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
            base = shader.prepare_base(raw_fr)

        for k, st in state.items():
            if st["ptr"] >= st["n_out"] or st["sched"][st["ptr"]][0] != pose_idx:
                continue
            style = st["cfg"]["style"]
            cached_base_style = None
            while st["ptr"] < st["n_out"] and st["sched"][st["ptr"]][0] == pose_idx:
                _, panel, scr_idx, q = st["sched"][st["ptr"]]
                t_out = st["ptr"] / 24.0
                prev_buf = st["prev_dec"].stdout.read(frame_bytes) if st["prev_dec"] is not None else b""

                # When prev_dec exists (Shackleton & Mars500), keep finalized drive frames (>11.5s) verbatim
                if (st["prev_dec"] is not None) and (t_out > 11.5) and (len(prev_buf) == frame_bytes):
                    st["enc"].stdin.write(prev_buf)
                    st["n_copied"] += 1
                else:
                    if cached_base_style is None or (style == "mars500" and t_out >= 194.0):
                        cached_base_style = shader.render_style(base, pose_idx, style, t_sec=t_out)
                    f32 = cached_base_style.copy()
                    if panel != 0:
                        f32 = composite_trilogy_screen(f32, panel, scr_idx, q, st["screens"], shader.hatch_bank[pose_idx % 12], rng)
                    if t_out <= 11.5:
                        f32 = apply_intro_if_active(f32, t_out, st["intro_bgr"], st["intro_a"])
                    f32 += rng.normal(0.0, 2.3, (H, W, 1)).astype(np.float32)
                    st["enc"].stdin.write(np.clip(f32, 0.0, 255.0).astype(np.uint8).tobytes())
                    st["n_rendered"] += 1
                st["ptr"] += 1

        if (pose_idx + 1) % 300 == 0:
            prog = ", ".join(f"{k}:{st['ptr']}/{st['n_out']}" for k, st in state.items())
            print(f"pose {pose_idx+1}/{N_SRC_POSES} -> {prog}", flush=True)

    dec.stdout.close()
    dec.wait()
    for k, st in state.items():
        if st["prev_dec"] is not None:
            st["prev_dec"].stdout.close()
            st["prev_dec"].wait()
        st["enc"].stdin.close()
        st["enc"].wait()
        assert st["ptr"] == st["n_out"], f"{k} incomplete: {st['ptr']} != {st['n_out']}"
        print(
            f"Head done: {k} wrote {st['ptr']} frames -> {st['head_out']} "
            f"(rendered={st['n_rendered']}, copied={st['n_copied']})"
        )

    # Render Protagonist Cubist elevator tails + final mux for all 3 videos
    for vcfg in VIDEOS:
        k = vcfg["key"]
        head_p = state[k]["head_out"]
        tail_p = os.path.join(TRI_DIR, f"tail_{k}.mp4")
        restyle_cubist_tail(vcfg, tail_p)

        hd = vcfg["head_dur"]
        tot = vcfg["total_dur"]
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
                "-t", f"{tot:.2f}", "-movflags", "+faststart", vcfg["out"], "-y",
            ],
            check=True,
        )
        print(f"MUXED FINAL: {vcfg['out']}")
        if os.path.isfile(state[k]["head_prev"]):
            os.remove(state[k]["head_prev"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if mode == "preview":
        run_preview()
    elif mode == "all":
        run_all()
    else:
        raise SystemExit(f"Unknown mode: {mode}")
