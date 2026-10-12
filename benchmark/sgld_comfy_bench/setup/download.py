"""Download what this card needs and link the ComfyUI files into COMFY_DIR/models.

Track A: ComfyUI single files (H3, LTX-2.5, Qwen-Image 2.1) plus the FastH3 / VDN directories.
Track B: the HF diffusers repos both engines load. Everything goes through HF_HUB_CACHE, so a
re-run skips finished files. HF_TOKEN comes from env.sh (HF_TOKEN_FILE); it is never printed.

Usage: download.py [--tracks A,B] [--skip fasth3,vdn] [--dry-run]
"""
import argparse
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

import models  # noqa: E402

KIT = pathlib.Path(__file__).resolve().parents[1]


def link(path, sub, name):
    dst = pathlib.Path(os.environ["COMFY_DIR"]) / "models" / sub / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(path)


def track_a(card, dry):
    for repo, rev, rel, sub in models.track_a_downloads(card):
        print(f"{'would fetch' if dry else 'fetch'} {repo}:{rel}", flush=True)
        if not dry:
            link(hf_hub_download(repo, rel, revision=rev), sub, pathlib.Path(rel).name)


def functional(card, dry):
    if card not in models.FUNCTIONAL_CARDS:
        return
    for repo, rev, rel, sub in models.FUNCTIONAL_DOWNLOADS:
        print(f"{'would fetch' if dry else 'fetch'} {repo}:{rel} (functional)", flush=True)
        if not dry:
            link(hf_hub_download(repo, rel, revision=rev), sub, pathlib.Path(rel).name)


def fasth3(dry):
    """FastH3 8-Step V2 for SGLD: transformer shards plus a model_index.json whose other components
    are null (ComfyUI supplies TE and VAEs)."""
    dst = pathlib.Path(models.FASTH3_DIR())
    print(f"fasth3 -> {dst} (~66 GB)", flush=True)
    if dry:
        return
    snap = pathlib.Path(snapshot_download(models.FASTH3_REPO, allow_patterns=["transformer/*"], max_workers=16))
    (dst / "transformer").mkdir(parents=True, exist_ok=True)
    for f in (snap / "transformer").iterdir():
        target = dst / "transformer" / f.name
        if not target.exists():
            target.symlink_to(f.resolve())
    (dst / "model_index.json").write_text((KIT / "setup/extras/model_index_fasth3.json").read_text())
    subprocess.run([sys.executable, str(KIT / "setup/extras/materialize-fasth3-v2-transformer.py"), str(dst)], check=True)


def vdn(dry):
    """VDN-H3: the OpenVDN repo plus a one-time 62 GB prefuse into WORKSPACE/models/VDN-H3-8nfe."""
    dst = pathlib.Path(models.VDN_DIR())
    print(f"vdn -> {dst} (~90 GB download + 62 GB prefuse)", flush=True)
    if dry:
        return
    snapshot_download(models.VDN_REPO, revision=models.VDN_REV, max_workers=16)
    if not (dst / "transformer").exists() or not any((dst / "transformer").glob("*.safetensors")):
        subprocess.run([sys.executable, str(KIT / "setup/extras/prefuse-vdn.py")], check=True)
    (dst / "model_index.json").write_text((KIT / "setup/extras/model_index_vdn.json").read_text())


def track_b(dry):
    for repo, rev, patterns in models.TRACK_B_REPOS:
        print(f"{'would fetch' if dry else 'fetch'} {repo}@{rev} {patterns or 'all'}", flush=True)
        if not dry:
            snapshot_download(repo, revision=rev, allow_patterns=patterns, max_workers=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="A,B")
    ap.add_argument("--skip", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    card = os.environ["CARD"]
    skip = set(filter(None, a.skip.split(",")))
    tracks = set(a.tracks.split(","))
    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN is empty (HF_TOKEN_FILE missing?); gated repos will fail", flush=True)
    if "A" in tracks:
        track_a(card, a.dry_run)
        functional(card, a.dry_run)
        if "fasth3" not in skip:
            fasth3(a.dry_run)
        if "vdn" not in skip:
            vdn(a.dry_run)
    if "B" in tracks:
        track_b(a.dry_run)
    print("DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
