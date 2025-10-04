# modal_app.py
import os, signal, subprocess, json, shutil
from datetime import datetime
from typing import Optional, List
import modal
from modal import Image, Volume, gpu
import argparse

app = modal.App("splats")


splats_wspace = Volume.from_name("workspace", create_if_missing=True)

 

image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .run_commands(
        # Prepare to add NVIDIA CUDA APT repo (for Nsight CLI tools)
        "apt-get update && apt-get install -y curl ca-certificates gnupg",
        "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
        "apt-get update",
    )
    .apt_install(
        # Common tools
        "git", "wget", "curl", "build-essential", "ccache", "gdb",
        # Rust toolchain + native build helpers for building Rust/Python extensions
        "cargo", "rustc", "pkg-config", "cmake", "ninja-build",
        # Kernel build deps
        "libnuma-dev",            # required by MSCCl++ and NUMA-aware components
        "rdma-core", "libibverbs-dev",  # optional RDMA/IB verbs support (non-fatal if unused)
        # Nsight CLI tools matching CUDA 12.8
        #"cuda-nsight-systems-12-8", "cuda-nsight-compute-12-8",
    )
    .uv_pip_install(
        # Ensure uv is present for runtime `python3 -m uv ...`
        "uv",
        # Build backends/tools for no-build-isolation flows
        "scikit-build-core",   # backend for sgl-kernel
        "setuptools-rust",     # backend for sgl-router
        "ninja",
        "setuptools",
        "wheel",
        "numpy",               # quiets PyTorch's numpy warning during configure
        # Runtime deps (kept)
        "pybase64",
        "huggingface_hub",
    )
)



GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "A100")






@app.local_entrypoint()
def sync_workspace(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", action="append", default=[], help="Additional directories to sync to workspace (defaults to ['3dgs', 'diff-gaussian-rasterization'] if none provided)")
    args, unknown = parser.parse_known_args(kwargs.values())
    dirs_to_sync = args.dir
    if not dirs_to_sync:
        dirs_to_sync = ['3dgs', 'diff-gaussian-rasterization']
    for src in dirs_to_sync:
        dest = f"/workspace/{os.path.basename(src)}"
        print(f"Syncing {src} -> {dest} ...")
        subprocess.run(["modal", "volume", "put", "workspace", src, dest], check=True)
        print(f"Done syncing {src}.")







@app.function(
    image=image,
    gpu=GPU,
    volumes={"/workspace": splats_wspace},
    timeout=30 * 60,
)
def buil_source():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-r", "/workspace/3dgs/requirements.txt"], cwd="/workspace", check=True)
    print("Installed requirements from 3dgs/requirements.txt")

    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-e", "/workspace/diff-gaussian-rasterization"], cwd="/workspace", check=True)
    print("Installed diff-gaussian-rasterization from source")

    print("Build complete")
