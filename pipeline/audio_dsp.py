#!/usr/bin/env python3
"""Assemble 4-Act Night-Chalk soundtrack + smoothly ducked voiceover master
(work/audio288/expedition_master.wav, 288.0s, -14.0 LUFS / -1.5 dBTP).
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import media_base as B

SR = 48000
DUR = 288.0
N_SAMPLES = int(round(DUR * SR))

MUSIC_DIR = os.path.join(B.ROOT, "music")
QUOTE_DIR = os.path.join(B.WORK, "expedition_assets", "audio")
OUT_DIR = os.path.join(B.WORK, "audio288")

# (quote_idx, film_start_s)
QUOTE_PLACEMENTS = [
    (1, 11.5),
    (2, 82.0),
    (3, 134.0),
    (4, 188.0),
    (5, 225.0),
    (6, 250.0),
]


def load_f32_stereo(path, start_s=0.0, dur_s=None, loudnorm=False):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.3f}"]
    cmd += ["-i", path]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16.0:TP=-2.0:LRA=8.0"]
    cmd += ["-ar", str(SR), "-ac", "2", "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    return arr


def save_wav(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = np.clip(arr, -0.999, 0.999).astype(np.float32).tobytes()
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
            "-c:a", "pcm_s16le", path, "-y",
        ],
        input=raw,
        check=True,
    )


def equal_power_xfade(arr, t0, t1, fade_in=True, n_total=N_SAMPLES):
    s0 = max(0, int(round(t0 * SR)))
    s1 = min(len(arr), int(round(t1 * SR)))
    if s1 <= s0:
        return
    u = np.linspace(0.0, 1.0, s1 - s0, dtype=np.float32)
    w = np.sin(0.5 * np.pi * u) if fade_in else np.cos(0.5 * np.pi * u)
    arr[s0:s1] *= w[:, None]


def build_music_bed():
    # Track specs aligned with 4.0s equal-power crossfades centered at 75.0s, 175.0s, 239.0s
    specs = [
        ("The Quiet Aftermath - Sir Cubworth.mp3", 0.0, 77.0, 0.0, None, 73.0),
        ("19th Floor - Bobby Richards.mp3", 0.0, 104.0, 73.0, 73.0, 173.0),
        ("Digital Ghosts - Unicorn Heads.mp3", 0.0, 68.0, 173.0, 173.0, 237.0),
        ("Stellar Wind - Unicorn Heads.mp3", 225.0, 51.0, 237.0, 237.0, None),
    ]
    bed = np.zeros((N_SAMPLES, 2), dtype=np.float32)
    for fname, src_st, dur_s, film_st, xf_in_st, xf_out_st in specs:
        seg = load_f32_stereo(os.path.join(MUSIC_DIR, fname), start_s=src_st, dur_s=dur_s, loudnorm=True)
        # Edge safety fades
        n_seg = len(seg)
        if film_st == 0.0:
            fi = min(int(1.2 * SR), n_seg)
            seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
        if xf_in_st is not None:
            loc0 = xf_in_st - film_st
            equal_power_xfade(seg, loc0, loc0 + 4.0, fade_in=True)
        if xf_out_st is not None:
            loc0 = xf_out_st - film_st
            equal_power_xfade(seg, loc0, loc0 + 4.0, fade_in=False)
        else:
            # Final 4.5s gentle outro fade ending by 287.5s
            fo_start = max(0, int(round((283.0 - film_st) * SR)))
            fo_end = min(n_seg, int(round((287.5 - film_st) * SR)))
            if fo_end > fo_start:
                u = np.linspace(0.0, 1.0, fo_end - fo_start, dtype=np.float32)
                seg[fo_start:fo_end] *= (0.5 * (1.0 + np.cos(np.pi * u)))[:, None]
                seg[fo_end:] = 0.0

        dst0 = int(round(film_st * SR))
        copy_len = min(n_seg, N_SAMPLES - dst0)
        bed[dst0 : dst0 + copy_len] += seg[:copy_len]
    return bed


def build_voice_track():
    voice = np.zeros((N_SAMPLES, 2), dtype=np.float32)
    timeline = []
    for q_idx, st_s in QUOTE_PLACEMENTS:
        qp = os.path.join(QUOTE_DIR, f"quote{q_idx}.wav")
        arr = load_f32_stereo(qp, start_s=0.0, dur_s=None, loudnorm=False)
        dst0 = int(round(st_s * SR))
        copy_len = min(len(arr), N_SAMPLES - dst0)
        voice[dst0 : dst0 + copy_len] += arr[:copy_len]
        dur = len(arr) / SR
        timeline.append((q_idx, st_s, st_s + dur))
        print(f"Placed quote{q_idx}.wav: {st_s:.2f}s -> {st_s + dur:.2f}s (dur {dur:.2f}s)")
    return voice, timeline


def compute_duck_gain(voice, min_gain=0.36, hop=960):
    """Compute smooth cosine-smoothed ducking gain curve driven by active voice energy."""
    mono = np.mean(np.abs(voice), axis=1)
    n_blocks = (len(mono) + hop - 1) // hop
    pad_len = n_blocks * hop - len(mono)
    if pad_len > 0:
        mono = np.pad(mono, (0, pad_len))
    rms = np.sqrt(np.mean(mono.reshape(n_blocks, hop) ** 2, axis=1))
    active = (rms > 0.0065).astype(np.float32)

    # Dilate slightly to bridge natural micro-breaths (<0.55s) without holding across >=1.5s pauses
    pad_blocks = int(round(0.28 / (hop / SR)))
    if pad_blocks > 0:
        k = np.ones(2 * pad_blocks + 1, dtype=np.float32)
        active = np.clip(np.convolve(active, k, mode="same"), 0.0, 1.0)

    # Asymmetric attack (0.45s) / release (0.65s) one-pole envelope over 20ms hops
    dt = hop / SR
    a_att = 1.0 - np.exp(-dt / 0.16)
    a_rel = 1.0 - np.exp(-dt / 0.28)
    env = np.zeros_like(active)
    cur = 0.0
    for i, target in enumerate(active):
        coef = a_att if target > cur else a_rel
        cur += coef * (target - cur)
        env[i] = cur

    gain_blocks = 1.0 - (1.0 - min_gain) * env
    # Upsample smoothly back to sample level
    x_blk = (np.arange(n_blocks, dtype=np.float64) + 0.5) * hop
    x_smp = np.arange(N_SAMPLES, dtype=np.float64)
    gain = np.interp(x_smp, x_blk, gain_blocks).astype(np.float32)
    return gain


def two_pass_loudnorm(in_wav, out_wav):
    r1 = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", in_wav,
            "-af", "loudnorm=I=-14.0:TP=-1.5:LRA=9.0:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(r"\{[^{}]+\}", r1.stderr, re.S)
    stats = json.loads(m.group(0))
    flt = (
        f"loudnorm=I=-14.0:TP=-1.5:LRA=9.0:linear=true:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", in_wav, "-af", flt, "-ar", str(SR), "-ac", "2",
            "-t", f"{DUR:.3f}", "-c:a", "pcm_s16le", out_wav, "-y",
        ],
        check=True,
    )
    r2 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", out_wav, "-af", "loudnorm=I=-14.0:TP=-1.5:LRA=9.0:print_format=summary", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    print("=== Final Audio Loudnorm Summary ===")
    for line in r2.stderr.splitlines():
        if any(k in line for k in ("Input Integrated", "Input True Peak", "Input LRA")):
            print(line.strip())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bed = build_music_bed()
    voice, _ = build_voice_track()
    duck = compute_duck_gain(voice, min_gain=0.36)
    mix = bed * duck[:, None] + voice * 0.96
    raw_out = os.path.join(OUT_DIR, "expedition_raw.wav")
    master_out = os.path.join(OUT_DIR, "expedition_master.wav")
    save_wav(mix, raw_out)
    two_pass_loudnorm(raw_out, master_out)
    print("Master wrote:", master_out)


if __name__ == "__main__":
    main()
