"""
Compare diffusers vs sglang checkpoint tensors for precision alignment.

Usage:
    python compare_checkpoints.py [--diffusers-dir /tmp/diffusers_ckpts] [--sglang-dir /tmp/sglang_ckpts]
    python compare_checkpoints.py --sort-by cosine          # sort results by cosine similarity (worst first)
    python compare_checkpoints.py --filter-step 0           # only compare step 0 checkpoints
    python compare_checkpoints.py --auto-reshape             # auto-fix shape mismatches (squeeze/permute)

Part of the sglang-diffusion-precision-alignment skill.
"""

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def normalize_shape(a: torch.Tensor, b: torch.Tensor):
    """Try to align tensor shapes between diffusers and sglang.

    Handles common mismatches:
    - [B, C, 1, H, W] vs [B, C, H, W]: squeeze singleton dim
    - [C, B, H, W] vs [B, C, H, W]: permute leading dims
    """
    # Squeeze singleton dim-2: [B, C, 1, H, W] → [B, C, H, W]
    if a.dim() == 5 and a.shape[2] == 1:
        a = a.squeeze(2)
    if b.dim() == 5 and b.shape[2] == 1:
        b = b.squeeze(2)
    # Permute leading dims: [C, B, H, W] → [B, C, H, W]
    if a.shape != b.shape and a.dim() == 4 and b.dim() == 4:
        if a.shape[0] == b.shape[1] and a.shape[1] == b.shape[0]:
            b = b.permute(1, 0, 2, 3)
    return a, b


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor):
    """Compare two tensors and print detailed statistics."""
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos_sim = cosine_sim(a, b)
    close_1e3 = torch.allclose(a_f, b_f, atol=1e-3, rtol=1e-3)
    close_1e2 = torch.allclose(a_f, b_f, atol=1e-2, rtol=1e-2)

    # Relative error: max_abs_diff / value range
    val_range = max(a_f.abs().max().item(), b_f.abs().max().item())
    max_rel = max_abs / val_range if val_range > 0 else 0.0

    status = "✅ MATCH" if close_1e3 else ("⚠️  CLOSE" if close_1e2 else "❌ DIFFER")

    print(f"\n{'='*70}")
    print(f"  {status}  {name}")
    print(f"{'='*70}")
    print(f"  Shape (diffusers): {list(a.shape)}   dtype: {a.dtype}")
    print(f"  Shape (sglang):    {list(b.shape)}   dtype: {b.dtype}")
    print(f"  max_abs_diff:      {max_abs:.6e}")
    print(f"  max_rel_diff:      {max_rel:.6e}  ({max_rel*100:.2f}%)")
    print(f"  mean_abs_diff:     {mean_abs:.6e}")
    print(f"  cosine_similarity: {cos_sim:.8f}")
    print(f"  allclose(1e-3):    {close_1e3}")
    print(f"  allclose(1e-2):    {close_1e2}")
    print(f"  diffusers range:   [{a_f.min().item():.4f}, {a_f.max().item():.4f}]  mean={a_f.mean().item():.4f}")
    print(f"  sglang   range:   [{b_f.min().item():.4f}, {b_f.max().item():.4f}]  mean={b_f.mean().item():.4f}")

    return {
        "name": name,
        "status": status,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": mean_abs,
        "cosine_similarity": cos_sim,
    }


def compare_scalars(name: str, a, b):
    """Compare scalar values with strict tolerance."""
    if isinstance(a, torch.Tensor):
        a = a.item()
    if isinstance(b, torch.Tensor):
        b = b.item()

    diff = abs(a - b)
    match = diff < 1e-6
    status = "✅ MATCH" if match else "❌ DIFFER"

    print(f"\n{'='*70}")
    print(f"  {status}  {name}")
    print(f"{'='*70}")
    print(f"  diffusers: {a}")
    print(f"  sglang:    {b}")
    print(f"  abs_diff:  {diff:.6e}")

    return {"name": name, "status": status, "diff": diff}


def _extract_step(name: str):
    """Extract step number from checkpoint name like 'cp4_step3_cond_out'."""
    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Compare diffusers vs sglang checkpoints")
    parser.add_argument("--diffusers-dir", default="/tmp/diffusers_ckpts", help="diffusers checkpoint dir")
    parser.add_argument("--sglang-dir", default="/tmp/sglang_ckpts", help="sglang checkpoint dir")
    parser.add_argument("--sort-by", choices=["cosine", "max_abs", "max_rel", "name"],
                        default="name", help="sort results (cosine sorts worst-first)")
    parser.add_argument("--filter-step", type=int, default=None,
                        help="only compare checkpoints for a specific step number")
    parser.add_argument("--auto-reshape", action="store_true",
                        help="auto-fix common shape mismatches (squeeze/permute)")
    args = parser.parse_args()

    d_dir = Path(args.diffusers_dir)
    s_dir = Path(args.sglang_dir)

    if not d_dir.exists() or not s_dir.exists():
        print(f"ERROR: checkpoint dirs not found: {d_dir} or {s_dir}")
        return

    # Collect checkpoint file names (intersection of both dirs)
    d_files = {f.stem for f in d_dir.glob("*.pt")}
    s_files = {f.stem for f in s_dir.glob("*.pt")}
    common = sorted(d_files & s_files)
    only_d = sorted(d_files - s_files)
    only_s = sorted(s_files - d_files)

    # Apply step filter
    if args.filter_step is not None:
        common = [n for n in common if _extract_step(n) == args.filter_step
                  or _extract_step(n) is None]

    print(f"\n{'#'*70}")
    print(f"  SGLang vs Diffusers Precision Comparison Report")
    print(f"{'#'*70}")
    print(f"  Diffusers dir: {d_dir}")
    print(f"  SGLang dir:    {s_dir}")
    print(f"  Common checkpoints: {len(common)}")
    if args.filter_step is not None:
        print(f"  Filter: step={args.filter_step}")
    if only_d:
        print(f"  Only in diffusers:  {only_d}")
    if only_s:
        print(f"  Only in sglang:     {only_s}")

    results = []
    for name in common:
        d_val = torch.load(d_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        s_val = torch.load(s_dir / f"{name}.pt", map_location="cpu", weights_only=True)

        if isinstance(d_val, torch.Tensor) and isinstance(s_val, torch.Tensor):
            # Scalar tensors: use strict scalar comparison
            if d_val.dim() == 0 and s_val.dim() == 0:
                r = compare_scalars(name, d_val, s_val)
                results.append(r)
            elif d_val.shape != s_val.shape:
                reshaped = False
                if args.auto_reshape:
                    d_val_n, s_val_n = normalize_shape(d_val, s_val)
                    if d_val_n.shape == s_val_n.shape:
                        print(f"\n  ℹ️  Auto-reshaped {name}: "
                              f"{list(d_val.shape)}+{list(s_val.shape)} → {list(d_val_n.shape)}")
                        r = compare_tensors(name, d_val_n, s_val_n)
                        results.append(r)
                        reshaped = True
                if not reshaped:
                    print(f"\n{'='*70}")
                    print(f"  ❌ SHAPE MISMATCH  {name}")
                    print(f"{'='*70}")
                    print(f"  diffusers: {list(d_val.shape)}")
                    print(f"  sglang:    {list(s_val.shape)}")
                    if not args.auto_reshape:
                        print(f"  hint: try --auto-reshape to attempt automatic shape alignment")
                    results.append({"name": name, "status": "❌ SHAPE MISMATCH"})
            else:
                r = compare_tensors(name, d_val, s_val)
                results.append(r)
        else:
            # Python scalar comparison
            r = compare_scalars(name, d_val, s_val)
            results.append(r)

    # Sort results
    if args.sort_by == "cosine":
        results.sort(key=lambda r: r.get("cosine_similarity", 2.0))
    elif args.sort_by == "max_abs":
        results.sort(key=lambda r: -r.get("max_abs_diff", 0.0))
    elif args.sort_by == "max_rel":
        results.sort(key=lambda r: -r.get("max_rel_diff", 0.0))

    # Summary report
    print(f"\n\n{'#'*70}")
    print(f"  SUMMARY" + (f"  (sorted by {args.sort_by})" if args.sort_by != "name" else ""))
    print(f"{'#'*70}")
    for r in results:
        line = f"  {r['status']:15s}  {r['name']}"
        if "cosine_similarity" in r:
            line += f"  cos={r['cosine_similarity']:.6f}"
        if "max_rel_diff" in r:
            line += f"  rel={r['max_rel_diff']*100:.2f}%"
        print(line)

    # Find first difference (only meaningful when sorted by name)
    first_differ = None
    for r in results:
        if "DIFFER" in r["status"] or "MISMATCH" in r["status"]:
            first_differ = r["name"]
            break
    if first_differ:
        print(f"\n  >>> First divergent checkpoint: {first_differ}")
        print(f"  >>> Investigate this stage first")
    else:
        print(f"\n  >>> All checkpoints match! Implementations are numerically aligned.")


if __name__ == "__main__":
    main()
