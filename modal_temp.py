# modal_app.py
import os, signal, subprocess, json, shutil
import tempfile
from pathlib import Path
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
def sync_workspace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", action="append", default=[], help="Additional directories to sync to workspace (defaults to ['3dgs', 'diff-gaussian-rasterization'] if none provided)")
    args, unknown = parser.parse_known_args()
    dirs_to_sync = args.dir
    if not dirs_to_sync:
        dirs_to_sync = ['3dgs', 'diff-gaussian-rasterization']
    for src in dirs_to_sync:
        dest = f"/{os.path.basename(src)}"
        print(f"Syncing {src} -> {dest} ...")
        # Delete remote dir if exists to allow overwrite
        subprocess.run(["modal", "volume", "rm", "--recursive", "workspace", dest], check=False)  # Ignore if not exists
        subprocess.run(["modal", "volume", "put", "workspace", src, dest], check=True)
        print(f"Done syncing {src}.")


@app.local_entrypoint()
def train_scene(
    scene: str,
    iters: int = 30000,
    init_count: int = 50000,
    lr_pos: float = 1e-2,
    lr_other: float = 1e-3,
    out_dir: str = "./out_local",
    images_dir: str = "images_8"
):
    import sys
    import os
    repo_root = os.path.dirname(__file__)
    train_src = os.path.join(repo_root, "3dgs")
    if train_src not in sys.path:
        sys.path.insert(0, train_src)
    from train_local import main as train_main

    scene_path = scene
    if not os.path.isdir(scene_path):
        candidate = os.path.join(repo_root, "scenes", scene)
        if os.path.isdir(candidate):
            scene_path = candidate

    argv = [
        "train_local",
        "--scene", scene_path,
        "--iters", str(iters),
        "--init_count", str(init_count),
        "--lr_pos", str(lr_pos),
        "--lr_other", str(lr_other),
        "--out", out_dir,
        "--images_dir", images_dir,
    ]

    prev_argv = sys.argv
    sys.argv = argv
    try:
        print(f"Starting local training for scene {scene}...")
        train_main()
    finally:
        sys.argv = prev_argv
    print("Training complete!")

    remote_dir = out_dir if out_dir.startswith("/") else f"/{Path(out_dir).name}"
    try:
        sync_outputs(remote_dir=remote_dir, local_dir=out_dir, volume_name="workspace")
    except subprocess.CalledProcessError as exc:
        print(f"Sync attempt from '{remote_dir}' failed (skipping): {exc}")


@app.local_entrypoint()
def sync_outputs(
    remote_dir: str = "/out_local",
    local_dir: str = "./out_local",
    volume_name: str = "workspace",
    overwrite: bool = True,
) -> None:
    """Sync a directory from a Modal volume down to the local filesystem."""
    remote_dir = remote_dir if remote_dir.startswith("/") else f"/{remote_dir}"
    local_path = Path(local_dir).expanduser().resolve()

    check_cmd = ["modal", "volume", "ls", volume_name, remote_dir]
    check = subprocess.run(check_cmd, capture_output=True, text=True)
    if check.returncode != 0:
        print(f"Remote path '{remote_dir}' not found on volume '{volume_name}'; skipping sync.")
        return

    if local_path.exists() and overwrite:
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    cmd = ["modal", "volume", "get", volume_name, remote_dir, str(local_path)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Synced {remote_dir} from volume '{volume_name}' to {local_path}.")


@app.local_entrypoint()
def test_sync_outputs(volume_name: str = "workspace") -> None:
    """Exercise sync_outputs by syncing a temporary directory containing a hello-world file."""
    remote_test_dir = "/sync_outputs_test"
    local_test_dir = Path("./test_synced").resolve()

    temp_dir = Path(tempfile.mkdtemp(prefix="modal_sync_src_"))
    test_file = temp_dir / "hello.txt"
    test_file.write_text("hello world\n")

    subprocess.run(
        ["modal", "volume", "rm", "--recursive", volume_name, remote_test_dir],
        check=False,
    )
    subprocess.run(
        ["modal", "volume", "put", volume_name, str(temp_dir), remote_test_dir],
        check=True,
    )

    if local_test_dir.exists():
        shutil.rmtree(local_test_dir)

    sync_outputs(remote_dir=remote_test_dir, local_dir=str(local_test_dir), volume_name=volume_name)

    synced_file = next(local_test_dir.rglob("hello.txt"), None)
    if not synced_file:
        raise RuntimeError("hello.txt not found after syncing; ensure modal CLI access is configured")

    print(f"Test sync succeeded; file located at {synced_file}")
