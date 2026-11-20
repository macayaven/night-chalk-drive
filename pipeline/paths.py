"""Shared path resolution for the Night-Chalk Drive pipeline."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DOCS_IMAGES_DIR = REPO_ROOT / "docs" / "images"

# Set NIGHT_CHALK_DATA_DIR to point at the folder containing raw MOV/music inputs and render caches.
MEDIA_ROOT = Path(os.environ.get("NIGHT_CHALK_DATA_DIR", str(REPO_ROOT))).expanduser().resolve()

ROOT = str(MEDIA_ROOT)
VID = str(MEDIA_ROOT / "videos")
WORK = str(MEDIA_ROOT / "work")
OUT = str(MEDIA_ROOT / "output")
MUSIC = str(MEDIA_ROOT / "music")

QUADS_FULL_DRIVE = str(DATA_DIR / "track_quads_full_drive.json")
QUADS_EXPEDITION = str(DATA_DIR / "track_quads_expedition.json")

CUBIST_REF = os.environ.get(
    "NIGHT_CHALK_CUBIST_REF",
    str(MEDIA_ROOT / "assets" / "cubist_reference.png"),
)
