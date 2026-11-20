"""Verification suite for pipeline imports, quad data, shader black gate, 6-DoF math, audio DSP, and repo hygiene."""
import importlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PIPELINE_MODULES = [
    "paths",
    "media_base",
    "baseline_v1_template_tracker",
    "quad_overlay",
    "display_mask",
    "planar_tracker_segments",
    "track_full_drive_display",
    "night_chalk_shader",
    "cubist_portrait",
    "fetch_archival_media",
    "audio_dsp",
    "build_audio_mix",
    "build_dashboard_screens",
    "build_trilogy_tails",
    "build_trilogy_films",
    "build_3d_finales",
    "build_article_assets",
]


@pytest.mark.parametrize("mod_name", PIPELINE_MODULES)
def test_pipeline_module_imports(mod_name):
    mod = importlib.import_module(f"pipeline.{mod_name}")
    assert mod is not None


def test_full_drive_quads_integrity():
    data = json.loads((REPO_ROOT / "data" / "track_quads_full_drive.json").read_text(encoding="utf-8"))
    assert data["n_poses"] == 3198
    assert data["final_visible_poses"] == 874
    quads = data["quads"]
    assert len(quads) == 3198

    visible_before_park = [q for q in quads[:2989] if q is not None]
    assert len(visible_before_park) == 874
    for q in visible_before_park:
        arr = np.asarray(q, dtype=np.float32)
        assert arr.shape == (4, 2)
        assert np.isfinite(arr).all()
        assert arr[[0, 3], 0].min() >= 150.0 and arr[[0, 3], 0].max() <= 740.0
        assert arr[:, 0].min() >= 150.0 and arr[:, 0].max() <= 1350.0
        assert arr[:, 1].min() >= 420.0 and arr[:, 1].max() <= 1000.0
        assert arr[1, 0] > arr[0, 0] and arr[3, 1] > arr[0, 1]


def test_night_chalk_shader_preserves_dark_sky():
    from pipeline.night_chalk_shader import ChalkRotoscopeEngine, H, W

    engine = ChalkRotoscopeEngine()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Dark night sky in top half
    frame[: H // 2, :, :] = (6, 6, 8)
    # Bright street highlight band in lower half
    frame[H // 2 + 40 : H // 2 + 160, 120:600, :] = (210, 195, 175)

    out = engine.render_pose(frame, pose_idx=0)
    assert out.shape == (H, W, 3)
    sky = out[: H // 2 - 40, :, :]
    assert float(sky.mean()) < 10.0
    assert float(sky.max()) < 15.0
    lit = out[H // 2 + 70 : H // 2 + 130, 200:520, :]
    assert float(lit.mean()) > 120.0


def test_rotation_gait_and_pinhole_projection():
    from pipeline import build_3d_finales as B3D

    R = B3D._rot_matrix(pitch_deg=1.5, yaw_deg=-2.0, roll_deg=0.75)
    assert R.shape == (3, 3)
    assert np.allclose(R.T @ R, np.eye(3, dtype=np.float32), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    gait = B3D._walk_kinematics(t_walk=1.25, step_hz=1.8, amp=1.0)
    assert len(gait) == 5
    assert all(np.isfinite(v) for v in gait)

    img = np.full((B3D.H, B3D.W, 3), 128, dtype=np.uint8)
    z_map = np.full((B3D.H, B3D.W), 1.5, dtype=np.float32)
    alpha = np.ones((B3D.H, B3D.W), dtype=np.float32)
    warped, w_alpha = B3D._project_3d_layer(
        img, z_map, vp=(360.0, 480.0), tz=0.25, pitch=0.5, yaw=-0.4, roll=0.2, alpha=alpha
    )
    assert warped.shape == (B3D.H, B3D.W, 3)
    assert w_alpha.shape == (B3D.H, B3D.W)
    assert float(w_alpha.min()) >= 0.0 and float(w_alpha.max()) <= 1.0


def test_audio_rms_ducking_attenuates_under_speech():
    from pipeline import audio_dsp as AD

    voice = np.zeros((AD.N_SAMPLES, 2), dtype=np.float32)
    # Place a clear speech-level pulse between 20s and 23s
    s0 = int(20.0 * AD.SR)
    s1 = int(23.0 * AD.SR)
    voice[s0:s1, :] = 0.08
    gain = AD.compute_duck_gain(voice, min_gain=0.36)
    assert gain.shape == (AD.N_SAMPLES,)
    assert float(gain[int(21.5 * AD.SR)]) < 0.42
    assert float(gain[int(5.0 * AD.SR)]) > 0.99


def test_repo_hygiene_no_local_paths_or_unicode_dashes():
    forbidden = [
        "/" + "Users/" + "carlos",
        "take-on-me-" + "helper",
        "helper-" + "handoff",
        "Mobile " + "Documents",
        "iCloud",
        "MANIFEST" + ".tsv",
        "Defect " + "PDF",
    ]
    banned_tells = ["thrilled", "delighted", "leverage", "delve", "tapestry", "robust", "seamless"]
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache"}
    self_path = Path(__file__).resolve()

    for path in REPO_ROOT.rglob("*"):
        if any(part in skip_dirs for part in path.parts) or not path.is_file():
            continue
        if path.resolve() == self_path or path.suffix not in {".py", ".md", ".toml", ".txt"}:
            continue
        txt = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in txt, f"Forbidden string {term!r} found in {path.relative_to(REPO_ROOT)}"
        if path.suffix == ".md":
            assert "\u2014" not in txt, f"Em-dash found in {path.relative_to(REPO_ROOT)}"
            assert "\u2013" not in txt, f"En-dash found in {path.relative_to(REPO_ROOT)}"
            for w in banned_tells:
                assert not re.search(rf"\b{w}\w*", txt, re.I), f"Banned word {w} in {path.relative_to(REPO_ROOT)}"
            in_fence = False
            for i, ln in enumerate(txt.splitlines(), 1):
                s = ln.strip()
                if s.startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence or s == "---" or re.fullmatch(r"\|[\s\-|]+\|", s):
                    continue
                prose = re.sub(r"`[^`]+`", "", s)
                assert "--" not in prose, f"Double hyphen '--' on line {i} of {path.relative_to(REPO_ROOT)}"
