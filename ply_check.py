import argparse
import json
import math
from pathlib import Path
import torch
from safetensors.torch import load_file
import sys

def parse_step(p: Path) -> int:
    s = p.stem.split("_")[-1]
    try:
        return int(s)
    except Exception:
        return -1

def tensor_stats(t: torch.Tensor) -> dict:
    finite = torch.isfinite(t).all().item()
    return {
        "shape": list(t.shape),
        "finite": bool(finite),
        "min": t.amin().item() if t.numel() > 0 else None,
        "max": t.amax().item() if t.numel() > 0 else None,
        "mean": t.mean().item() if t.numel() > 0 else None,
    }

def summarize_ckpt(p: Path, sample: int) -> dict:
    st = load_file(str(p))
    means = st["means"]
    scales = st["scales"]
    quats = st["quats"]
    opac = st["opac"]
    colors = st["colors"]
    n = means.shape[0]
    idxs = torch.linspace(0, max(n - 1, 0), steps=min(sample, n)).long()
    qn = torch.linalg.norm(quats, dim=1)
    s = {
        "file": str(p),
        "step": parse_step(p),
        "count": int(n),
        "means_min": means.amin(dim=0).tolist(),
        "means_max": means.amax(dim=0).tolist(),
        "scales_min": scales.amin(dim=0).tolist(),
        "scales_max": scales.amax(dim=0).tolist(),
        "quat_norm_min": float(qn.amin().item()),
        "quat_norm_max": float(qn.amax().item()),
        "opacity_min": float(opac.amin().item()),
        "opacity_max": float(opac.amax().item()),
        "colors_min": colors.amin(dim=0).tolist(),
        "colors_max": colors.amax(dim=0).tolist(),
        "means_stats": tensor_stats(means),
        "scales_stats": tensor_stats(scales),
        "quats_stats": tensor_stats(quats),
        "opac_stats": tensor_stats(opac),
        "colors_stats": tensor_stats(colors),
        "samples": [],
    }
    for i in idxs.tolist():
        s["samples"].append({
            "idx": int(i),
            "xyz": means[i].tolist(),
            "scale": scales[i].tolist(),
            "quat": quats[i].tolist(),
            "opacity": float(opac[i].item() if opac.dim() == 2 else opac[i].tolist()),
            "color": colors[i].tolist(),
        })
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ply", nargs="*", default=None)
    ap.add_argument("--scan_limit", type=int, default=0, help="max vertices to scan per PLY (0 = all)")
    args = ap.parse_args()
    if args.ckpt_dir:
        d = Path(args.ckpt_dir)
        files = sorted(d.glob("state_*.safetensors"), key=parse_step)
        if args.limit and args.limit > 0:
            files = files[: args.limit]
        out = [summarize_ckpt(p, args.sample) for p in files]
        print(json.dumps(out, indent=2))
    if args.ply:
        for ply in args.ply:
            p = Path(ply)
            with p.open("r") as f:
                # header
                fmt = None
                props = []
                n = None
                while True:
                    line = f.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s.startswith("format "):
                        fmt = s.split()[1]
                    if s.startswith("element vertex"):
                        n = int(s.split()[2])
                    if s.startswith("property "):
                        props.append(s.split()[-1])
                    if s == "end_header":
                        break
                ok_hdr = props[:14] == ["x","y","z","red","green","blue","scale_x","scale_y","scale_z","qw","qx","qy","qz","opacity"]
                if fmt != "ascii" or n is None:
                    print(json.dumps({"file": str(p), "ok": False, "reason": "unsupported format or missing vertex count"}))
                    continue
                limit = args.scan_limit if args.scan_limit and args.scan_limit > 0 else n
                read = 0
                bad = 0
                nan = 0
                import numpy as np
                xyz_min = [math.inf]*3; xyz_max = [-math.inf]*3
                rgb_min = [255]*3; rgb_max = [0]*3
                scale_min = [math.inf]*3; scale_max = [-math.inf]*3
                op_min = math.inf; op_max = -math.inf
                qn_min = math.inf; qn_max = -math.inf
                while read < limit:
                    line = f.readline()
                    if not line:
                        break
                    toks = line.strip().split()
                    if len(toks) != 14:
                        bad += 1; read += 1; continue
                    try:
                        x,y,z = map(float, toks[0:3])
                        r,g,b = map(int, toks[3:6])
                        sx,sy,sz = map(float, toks[6:9])
                        qw,qx,qy,qz = map(float, toks[9:13])
                        op = float(toks[13])
                    except Exception:
                        bad += 1; read += 1; continue
                    vals = [x,y,z,sx,sy,sz,qw,qx,qy,qz,op]
                    if any((math.isnan(v) or math.isinf(v)) for v in vals):
                        nan += 1; read += 1; continue
                    xyz_min = [min(xyz_min[j], v) for j,v in enumerate((x,y,z))]
                    xyz_max = [max(xyz_max[j], v) for j,v in enumerate((x,y,z))]
                    rgb_min = [min(rgb_min[j], v) for j,v in enumerate((r,g,b))]
                    rgb_max = [max(rgb_max[j], v) for j,v in enumerate((r,g,b))]
                    scale_min = [min(scale_min[j], v) for j,v in enumerate((sx,sy,sz))]
                    scale_max = [max(scale_max[j], v) for j,v in enumerate((sx,sy,sz))]
                    qn = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
                    qn_min = min(qn_min, qn); qn_max = max(qn_max, qn)
                    op_min = min(op_min, op); op_max = max(op_max, op)
                    read += 1
                print(json.dumps({
                    "file": str(p),
                    "format": fmt,
                    "header_ok": ok_hdr,
                    "vertex_count": n,
                    "scanned": read,
                    "bad": bad,
                    "nan": nan,
                    "xyz_min": xyz_min,
                    "xyz_max": xyz_max,
                    "rgb_min": rgb_min,
                    "rgb_max": rgb_max,
                    "scale_min": scale_min,
                    "scale_max": scale_max,
                    "quat_norm_min": qn_min,
                    "quat_norm_max": qn_max,
                    "opacity_min": op_min,
                    "opacity_max": op_max,
                }, indent=2))

if __name__ == "__main__":
    main()


