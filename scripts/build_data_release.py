"""Build a data release package: random 1/7 of all .npz dumps from
main/ and streaming/, preserving session/obs_/cal_ structure."""

import os
import random
import shutil
import tarfile
from pathlib import Path

ROOT = Path("/home/ikaros/projects/ay-121/data/lab04")
SOURCES = ["main", "streaming"]
OUT_DIR = Path("/home/ikaros/projects/ay-121/releases/lab04_release_1of7")
TARBALL = Path("/home/ikaros/projects/ay-121/releases/lab04_release_1of7.tar.gz")
FRACTION_DENOM = 7
SEED = 20260429

random.seed(SEED)

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

manifest_lines = []
totals = {}
selected_counts = {}

for src in SOURCES:
    src_root = ROOT / src
    all_files = sorted(str(p) for p in src_root.rglob("*.npz"))
    totals[src] = len(all_files)
    n_select = len(all_files) // FRACTION_DENOM
    chosen = random.sample(all_files, n_select)
    selected_counts[src] = len(chosen)
    print(f"{src}: total={len(all_files)}, selected={len(chosen)}")

    for f in chosen:
        rel = Path(f).relative_to(ROOT)
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(f, dst) if False else shutil.copy2(f, dst)
        manifest_lines.append(str(rel))

manifest_path = OUT_DIR / "MANIFEST.txt"
with manifest_path.open("w") as fh:
    fh.write("# Lab 04 data release\n\n")
    for line in sorted(manifest_lines):
        fh.write(line + "\n")

print(f"\nWriting tarball to {TARBALL} ...")
TARBALL.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(TARBALL, "w:gz") as tf:
    tf.add(OUT_DIR, arcname=OUT_DIR.name)

print(f"Done: {TARBALL} ({TARBALL.stat().st_size / 1e6:.1f} MB)")
