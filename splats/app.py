import json
import os
import subprocess
import sys
from pathlib import Path
import shutil
import tarfile
import time

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(__file__))

import modal
from modal import Image, Volume, gpu

DEFAULT_APP_NAME = "splats"
DEFAULT_VOLUME_NAME = "splats-workspace"
BASE_MOUNT = "/vol"


app = modal.App(os.environ.get("MODAL_APP_NAME", DEFAULT_APP_NAME))

shared_volume = Volume.from_name(
    os.environ.get("MODAL_VOLUME_NAME", DEFAULT_VOLUME_NAME),
    create_if_missing=True,
)


image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .run_commands(
        "apt-get update && apt-get install -y curl ca-certificates gnupg",
        "curl -fsSL -o //tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
        "apt-get update",
    )
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "ccache",
        "gdb",
        "cargo",
        "rustc",
        "pkg-config",
        "cmake",
        "ninja-build",
        "libnuma-dev",
        "rdma-core",
        "libibverbs-dev",
        "cuda-nsight-systems-12-8",
        "cuda-nsight-compute-12-8",
        "libgl1",
        "libglib2.0-0",
        "libxrender1",
        "libxext6",
        "libxi6",
    )
    .uv_pip_install(
        "uv",
        "scikit-build-core",
        "setuptools-rust",
        "ninja",
        "setuptools",
        "wheel",
        "numpy",
    )
)


GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU = {
    "L4": gpu.L4(),
    "L40S": gpu.L40S(),
    "A100": gpu.A100(),
    "H100": gpu.H100(),
}.get(GPU_KIND, gpu.L4())


def _volume_path(*parts: str) -> Path:
    return Path(BASE_MOUNT, *parts)


 


 



def _clean_local(paths: list[str]) -> None:
    for rel in paths:
        p = Path(rel)
        if not p.exists():
            continue
        for d in p.rglob("__pycache__"):
            shutil.rmtree(d, ignore_errors=True)
        for f in p.rglob("*.pyc"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass


@app.function(image=image, gpu=None, volumes={BASE_MOUNT: shared_volume})
def clear_volume(paths: list[str]) -> str:
    cleared: list[str] = []
    for rel in paths:
        p = _volume_path(rel)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                try:
                    p.unlink()
                except Exception:
                    pass
        cleared.append(str(p))
    return json.dumps(cleared)


@app.function(image=image, gpu=None, volumes={BASE_MOUNT: shared_volume})
def extract_archive(remote_tgz: str) -> str:
    tgz_path = _volume_path(remote_tgz)
    if not tgz_path.exists():
        raise FileNotFoundError(str(tgz_path))
    with tarfile.open(tgz_path, "r:gz") as t:
        t.extractall(path=Path(BASE_MOUNT))
    try:
        tgz_path.unlink()
    except Exception:
        pass
    return str(tgz_path)


def _make_tar(src_dir: str, out_path: str) -> None:
    excludes = ["__pycache__", ".pyc"]
    cmd = [
        "bash",
        "-lc",
        f"tar -C {os.getcwd()} --exclude='__pycache__' --exclude='*.pyc' -czf {out_path} {src_dir}",
    ]
    subprocess.run(cmd, check=True)


 


 



@app.local_entrypoint()
def sync_workspace() -> None:
    volume_name = os.environ.get("MODAL_VOLUME_NAME", DEFAULT_VOLUME_NAME)
    _clean_local(["splats", "diff-gaussian-rasterization"]) 
    print(f"Clearing remote paths in volume '{volume_name}' ...")
    print(clear_volume.remote(["uploads"]))
    print(clear_volume.remote(["splats", "diff-gaussian-rasterization"]))
    ts = str(int(time.time()))
    splats_tgz = f"/tmp/splats-{ts}.tar.gz"
    dgr_tgz = f"/tmp/dgr-{ts}.tar.gz"
    splats_remote = f"uploads/splats-{ts}.tar.gz"
    dgr_remote = f"uploads/dgr-{ts}.tar.gz"
    try:
        for p in Path("/tmp").glob("splats-*.tar.gz"):
            p.unlink()
        for p in Path("/tmp").glob("dgr-*.tar.gz"):
            p.unlink()
    except Exception:
        pass
    _make_tar("splats", splats_tgz)
    if Path("diff-gaussian-rasterization").exists():
        _make_tar("diff-gaussian-rasterization", dgr_tgz)
    print(f"Uploading {splats_tgz} -> {splats_remote} ...")
    subprocess.run(["modal", "volume", "put", volume_name, splats_tgz, splats_remote], check=True)
    if Path("diff-gaussian-rasterization").exists():
        print(f"Uploading {dgr_tgz} -> {dgr_remote} ...")
        subprocess.run(["modal", "volume", "put", volume_name, dgr_tgz, dgr_remote], check=True)
    print("Extracting remote archives ...")
    print(extract_archive.remote(splats_remote))
    if Path("diff-gaussian-rasterization").exists():
        print(extract_archive.remote(dgr_remote))
    try:
        Path(splats_tgz).unlink()
    except Exception:
        pass
    if Path("diff-gaussian-rasterization").exists():
        try:
            Path(dgr_tgz).unlink()
        except Exception:
            pass
    try:
        for p in Path("/tmp").glob("splats-*.tar.gz"):
            p.unlink()
        for p in Path("/tmp").glob("dgr-*.tar.gz"):
            p.unlink()
    except Exception:
        pass
    print("Workspace sync complete.")



def _detect_sm() -> str:
    import re

    try:
        output = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi -q -d COMPUTE || true"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r"Compute\s+Capability\s*:\s*([0-9]+)\.([0-9]+)", output)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    except Exception:
        pass
    return "8.9"



def _arch_lists(sm_csv: str | None) -> tuple[str, str]:
    sm_csv = sm_csv or _detect_sm()
    entries = [value.strip() for value in sm_csv.split(";") if value.strip()]
    torch_list = ";".join(entries)
    flashinfer_list = ";".join(f"{int(float(value) * 10)}" for value in entries)
    return torch_list, flashinfer_list


 


@app.function(image=image, gpu=GPU, volumes={BASE_MOUNT: shared_volume}, timeout=2 * 60 * 60)
def build_all(sm_targets: str = "") -> str:
    torch_arch, _ = _arch_lists(sm_targets)
    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch
    ws_root = _volume_path("")
    splats_dir = ws_root / "splats"
    dgr_dir = ws_root / "diff-gaussian-rasterization"
    req = splats_dir / "requirements.txt"
    if req.exists():
        subprocess.run(["bash", "-lc", f"python -m pip install -r '{req}'"], check=True)
    if dgr_dir.exists():
        subprocess.run(["bash", "-lc", f"python -m pip install -v '{dgr_dir}'"], check=True)
 