# modal_app.py
import os, signal, subprocess, json, shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import modal
from modal import Image, Volume, gpu
import argparse
import zipfile
import io

app = modal.App("splats")


splats_wspace = Volume.from_name("workspace", create_if_missing=True)

 

image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .env({"HF_HOME": "/workspace/hf"})
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
        # OpenGL libraries for headless rendering
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1",
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
    parser.add_argument("--dir", action="append", default=[], help="Additional directories to sync to workspace (defaults to ['dreamgaussian', 'diff-gaussian-rasterization'] if none provided)")
    args, unknown = parser.parse_known_args()
    dirs_to_sync = args.dir
    if not dirs_to_sync:
        dirs_to_sync = ['zero123plus']
        #dirs_to_sync = ['images']
    for src in dirs_to_sync:
        dest = f"/{os.path.basename(src)}"
        print(f"Syncing {src} -> {dest} ...")
        # Delete remote dir if exists to allow overwrite
        subprocess.run(["modal", "volume", "rm", "--recursive", "workspace", dest], check=False)  # Ignore if not exists
        subprocess.run(["modal", "volume", "put", "workspace", src, dest], check=True)
        print(f"Done syncing {src}.")




@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def zip_remote_dir(remote_dir: str = "/workspace/out_local") -> Optional[bytes]:
    if not os.path.exists(remote_dir):
        print(f"Remote path '{remote_dir}' not found")
        return None

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(remote_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, remote_dir)
                zipf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.read()


@app.local_entrypoint()
def sync_outputs(local_dir: str = "./out_local"):
    local_path = Path(local_dir).expanduser().resolve()
    
    print(f"Downloading /workspace/out_local to {local_path}")
    
    zip_data = zip_remote_dir.remote()
    
    if not zip_data:
        print("No data found at /workspace/out_local")
        return
    
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zipf:
        zipf.extractall(local_path)
    
    print(f"Downloaded to {local_path}")



@app.function(
     image=image,
     gpu="A100",
     volumes={"/workspace": splats_wspace},
     timeout=1800
)
def install_zero():
    import sys, os, subprocess, pathlib
    sys.path.insert(0, "/workspace/zero123plus")

    venv_path = "/workspace/venv_zero"
    py_bin = "/opt/conda/bin/python"
    if not os.path.exists(venv_path):
        subprocess.run([py_bin, "-m", "venv", venv_path], check=True)
    venv_env = os.environ.copy()
    venv_env["PATH"] = f"{venv_path}/bin:" + venv_env.get("PATH", "")
    venv_env["VIRTUAL_ENV"] = venv_path

    pyver = subprocess.check_output([py_bin, "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"]).decode().strip()
    site_packages = f"{venv_path}/lib/python{pyver}/site-packages"
    pathlib.Path(site_packages).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(site_packages, "_conda_site.pth"), "w") as f:
        f.write(f"/opt/conda/lib/python{pyver}/site-packages\n")

    os.chdir("/workspace/zero123plus")
    subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], env=venv_env, check=True)


    build_env = venv_env.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "8.0"
    splats_wspace.commit()



@app.function(
    image=image,
    gpu="A100",
    volumes={"/workspace": splats_wspace},
    timeout=1800
)
def inference():

    
    import sys, os, subprocess, pathlib
    sys.path.insert(0, "/workspace/zero123plus")

    venv_path = "/workspace/venv_zero"
    py_bin = "/opt/conda/bin/python"
    if not os.path.exists(venv_path):
        subprocess.run([py_bin, "-m", "venv", venv_path], check=True)
    venv_env = os.environ.copy()
    venv_env["PATH"] = f"{venv_path}/bin:" + venv_env.get("PATH", "")
    venv_env["VIRTUAL_ENV"] = venv_path
    
    os.chdir("/workspace/zero123plus")
    
    py = f"{venv_path}/bin/python"
    
    subprocess.run([py, "example.py"], env=venv_env, check=True)
    # Removed sync_outputs.remote() - now handled in local main


@app.local_entrypoint()
def run_inference():
    inference.remote()
    sync_outputs()


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def shell():
    import sys
    import subprocess
    subprocess.call(["/bin/bash", "-c", "cd /workspace && /bin/bash"], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

