#!/usr/bin/env python3
"""Prepare 12 trilogy quotes and mix/master the 3 standalone trilogy soundtracks:
  1. work/audio288/trilogy_01_nansen_master.wav     (280.00s, -14.0 LUFS)
  2. work/audio288/trilogy_02_shackleton_master.wav (240.00s, -14.0 LUFS)
  3. work/audio288/trilogy_03_mars500_master.wav    (255.00s, -14.0 LUFS)
"""
import os
import shutil
import sys

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import audio_dsp as BEA, fetch_archival_media as FEA, media_base as B

SR = BEA.SR
RAW_DIR = FEA.RAW_DIR
EXP_QUOTE_DIR = FEA.AUDIO_DIR
TRI_AUDIO_DIR = os.path.join(B.WORK, "trilogy_assets", "audio")
MUSIC_DIR = BEA.MUSIC_DIR


def prepare_trilogy_quotes():
    os.makedirs(TRI_AUDIO_DIR, exist_ok=True)
    tmp = os.path.join(TRI_AUDIO_DIR, "_parts")
    os.makedirs(tmp, exist_ok=True)

    # --- Video 1: Nansen ---
    shutil.copy2(os.path.join(EXP_QUOTE_DIR, "quote1.wav"), os.path.join(TRI_AUDIO_DIR, "nansen_q1.wav"))

    nq2 = os.path.join(tmp, "nq2.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "nansen_08.mp3"), 97.90, 131.65 - 97.90, nq2)
    FEA.concat_with_silences([(nq2, 0.0)], os.path.join(TRI_AUDIO_DIR, "nansen_q2.wav"))

    nq3a, nq3b = os.path.join(tmp, "nq3a.wav"), os.path.join(tmp, "nq3b.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "nansen_11.mp3"), 20.40, 35.40 - 20.40, nq3a)
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "nansen_11.mp3"), 55.28, 59.95 - 55.28, nq3b)
    FEA.concat_with_silences([(nq3a, 0.35), (nq3b, 0.0)], os.path.join(TRI_AUDIO_DIR, "nansen_q3.wav"))

    nq4 = os.path.join(tmp, "nq4.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "nansen_23.mp3"), 609.30, 621.75 - 609.30, nq4)
    FEA.concat_with_silences([(nq4, 0.0)], os.path.join(TRI_AUDIO_DIR, "nansen_q4.wav"))

    # --- Video 2: Shackleton ---
    shutil.copy2(os.path.join(EXP_QUOTE_DIR, "quote3.wav"), os.path.join(TRI_AUDIO_DIR, "shackleton_q1.wav"))

    sq2 = os.path.join(tmp, "sq2.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "shackleton_11.mp3"), 502.60, 532.20 - 502.60, sq2)
    FEA.concat_with_silences([(sq2, 0.0)], os.path.join(TRI_AUDIO_DIR, "shackleton_q2.wav"))

    sq3a, sq3b = os.path.join(tmp, "sq3a.wav"), os.path.join(tmp, "sq3b.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "shackleton_32.mp3"), 63.30, 78.90 - 63.30, sq3a)
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "shackleton_32.mp3"), 445.15, 458.20 - 445.15, sq3b)
    FEA.concat_with_silences([(sq3a, 0.40), (sq3b, 0.0)], os.path.join(TRI_AUDIO_DIR, "shackleton_q3.wav"))

    sq4 = os.path.join(tmp, "sq4.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "shackleton_1910_voice.mp3"), 165.50, 203.60 - 165.50, sq4, vintage_1910=True)
    FEA.concat_with_silences([(sq4, 0.0)], os.path.join(TRI_AUDIO_DIR, "shackleton_q4.wav"))

    # --- Video 3: Mars500 ---
    shutil.copy2(os.path.join(EXP_QUOTE_DIR, "quote4.wav"), os.path.join(TRI_AUDIO_DIR, "mars500_q1.wav"))

    mq2 = os.path.join(tmp, "mq2.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "mars500_romain_surprise.mp3"), 3.45, 44.30 - 3.45, mq2)
    FEA.concat_with_silences([(mq2, 0.0)], os.path.join(TRI_AUDIO_DIR, "mars500_q2.wav"))

    shutil.copy2(os.path.join(EXP_QUOTE_DIR, "quote5.wav"), os.path.join(TRI_AUDIO_DIR, "mars500_q3.wav"))

    mq4 = os.path.join(tmp, "mq4.wav")
    FEA.extract_clean_slice(os.path.join(RAW_DIR, "mars500_diego.mp3"), 89.60, 123.15 - 89.60, mq4)
    FEA.concat_with_silences([(mq4, 0.0)], os.path.join(TRI_AUDIO_DIR, "mars500_q4.wav"))

    for k in (
        "nansen_q1", "nansen_q2", "nansen_q3", "nansen_q4",
        "shackleton_q1", "shackleton_q2", "shackleton_q3", "shackleton_q4",
        "mars500_q1", "mars500_q2", "mars500_q3", "mars500_q4",
    ):
        p = os.path.join(TRI_AUDIO_DIR, f"{k}.wav")
        print(f"{k:16s}: {FEA.wav_dur(p):5.2f}s")


def build_nansen_bed(dur=280.0):
    n_s = int(round(dur * SR))
    seg = BEA.load_f32_stereo(os.path.join(MUSIC_DIR, "Stellar Wind - Unicorn Heads.mp3"), 0.0, dur, loudnorm=True)
    bed = np.zeros((n_s, 2), dtype=np.float32)
    copy_n = min(len(seg), n_s)
    bed[:copy_n] = seg[:copy_n]
    # 1.5s fade-in, 15s cosine fade-out 265..280s (04:25..04:40)
    fi = min(int(1.5 * SR), n_s)
    bed[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    fo0, fo1 = int(265.0 * SR), n_s
    u = np.linspace(0.0, 1.0, fo1 - fo0, dtype=np.float32)
    bed[fo0:fo1] *= (0.5 * (1.0 + np.cos(np.pi * u)))[:, None]
    return bed


def build_shackleton_bed(dur=240.0):
    """Extend 19th Floor (120.39s) seamlessly to 240.0s with 3.0s equal-power crossfades."""
    src = os.path.join(MUSIC_DIR, "19th Floor - Bobby Richards.mp3")
    n_s = int(round(dur * SR))
    bed = np.zeros((n_s, 2), dtype=np.float32)

    # Seg A: 0..108.0s placed at 0.0..108.0s (fade-out 105..108)
    sA = BEA.load_f32_stereo(src, 0.0, 108.0, loudnorm=True)
    fi = min(int(1.5 * SR), len(sA))
    sA[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    BEA.equal_power_xfade(sA, 105.0, 108.0, fade_in=False)
    bed[:len(sA)] += sA

    # Seg B: 32..108.0s (76.0s) placed at 105.0..181.0s (xfade in 0..3, out 73..76)
    sB = BEA.load_f32_stereo(src, 32.0, 76.0, loudnorm=True)
    BEA.equal_power_xfade(sB, 0.0, 3.0, fade_in=True)
    BEA.equal_power_xfade(sB, 73.0, 76.0, fade_in=False)
    dB = int(105.0 * SR)
    bed[dB : dB + len(sB)] += sB

    # Seg C: 32..94.0s (62.0s) placed at 178.0..240.0s (xfade in 0..3, cosine fade-out 226..240s)
    sC = BEA.load_f32_stereo(src, 32.0, 62.0, loudnorm=True)
    BEA.equal_power_xfade(sC, 0.0, 3.0, fade_in=True)
    dC = int(178.0 * SR)
    c_len = min(len(sC), n_s - dC)
    bed[dC : dC + c_len] += sC[:c_len]

    fo0, fo1 = int(226.0 * SR), n_s
    u = np.linspace(0.0, 1.0, fo1 - fo0, dtype=np.float32)
    bed[fo0:fo1] *= (0.5 * (1.0 + np.cos(np.pi * u)))[:, None]
    return bed


def build_mars500_bed(dur=255.0):
    n_s = int(round(dur * SR))
    seg = BEA.load_f32_stereo(os.path.join(MUSIC_DIR, "Digital Ghosts - Unicorn Heads.mp3"), 0.0, dur, loudnorm=True)
    bed = np.zeros((n_s, 2), dtype=np.float32)
    copy_n = min(len(seg), n_s)
    bed[:copy_n] = seg[:copy_n]
    fi = min(int(1.5 * SR), n_s)
    bed[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    fo0, fo1 = int(240.0 * SR), n_s
    u = np.linspace(0.0, 1.0, fo1 - fo0, dtype=np.float32)
    bed[fo0:fo1] *= (0.5 * (1.0 + np.cos(np.pi * u)))[:, None]
    return bed


def mix_trilogy_track(key, bed, placements, dur):
    n_s = int(round(dur * SR))
    voice = np.zeros((n_s, 2), dtype=np.float32)
    for q_name, st_s in placements:
        arr = BEA.load_f32_stereo(os.path.join(TRI_AUDIO_DIR, f"{q_name}.wav"), 0.0, None, loudnorm=False)
        dst0 = int(round(st_s * SR))
        copy_len = min(len(arr), n_s - dst0)
        voice[dst0 : dst0 + copy_len] += arr[:copy_len]
        print(f"  [{key}] {q_name}: {st_s:.2f}s -> {st_s + copy_len / SR:.2f}s")

    # Compute duck gain cleanly trimmed to n_s
    mono = np.mean(np.abs(voice), axis=1)
    hop = 960
    n_blocks = (len(mono) + hop - 1) // hop
    pad_len = n_blocks * hop - len(mono)
    if pad_len > 0:
        mono = np.pad(mono, (0, pad_len))
    rms = np.sqrt(np.mean(mono.reshape(n_blocks, hop) ** 2, axis=1))
    active = (rms > 0.0065).astype(np.float32)
    pad_blocks = int(round(0.28 / (hop / SR)))
    active = np.clip(np.convolve(active, np.ones(2 * pad_blocks + 1, dtype=np.float32), mode="same"), 0.0, 1.0)

    dt = hop / SR
    a_att = 1.0 - np.exp(-dt / 0.16)
    a_rel = 1.0 - np.exp(-dt / 0.28)
    env = np.zeros_like(active)
    cur = 0.0
    for i, target in enumerate(active):
        coef = a_att if target > cur else a_rel
        cur += coef * (target - cur)
        env[i] = cur
    gain_blocks = 1.0 - (1.0 - 0.35) * env
    x_blk = (np.arange(n_blocks, dtype=np.float64) + 0.5) * hop
    gain = np.interp(np.arange(n_s, dtype=np.float64), x_blk, gain_blocks).astype(np.float32)

    mix = bed * gain[:, None] + voice * 0.96
    raw_wav = os.path.join(BEA.OUT_DIR, f"{key}_raw.wav")
    master_wav = os.path.join(BEA.OUT_DIR, f"{key}_master.wav")
    BEA.save_wav(mix, raw_wav)

    # Temporary override of BEA.DUR for two_pass_loudnorm
    old_dur = BEA.DUR
    BEA.DUR = dur
    BEA.two_pass_loudnorm(raw_wav, master_wav)
    BEA.DUR = old_dur
    print(f"Master wrote: {master_wav} (dur={dur:.2f}s)")


def main():
    prepare_trilogy_quotes()
    mix_trilogy_track(
        "trilogy_01_nansen",
        build_nansen_bed(280.0),
        [("nansen_q1", 35.0), ("nansen_q2", 105.0), ("nansen_q3", 180.0), ("nansen_q4", 232.0)],
        280.0,
    )
    mix_trilogy_track(
        "trilogy_02_shackleton",
        build_shackleton_bed(240.0),
        [("shackleton_q1", 25.0), ("shackleton_q2", 90.0), ("shackleton_q3", 155.0), ("shackleton_q4", 192.0)],
        240.0,
    )
    mix_trilogy_track(
        "trilogy_03_mars500",
        build_mars500_bed(255.0),
        [("mars500_q1", 30.0), ("mars500_q2", 95.0), ("mars500_q3", 162.0), ("mars500_q4", 202.0)],
        255.0,
    )


if __name__ == "__main__":
    main()
