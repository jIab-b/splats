import json
import os
import subprocess
from pathlib import Path

import modal
from modal import Image, Volume, gpu
from nvstack import set_build_env, build_3dgs, initialize_splats_from_gltf, export_splats_ply

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
        "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
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
        "pybase64",
        "huggingface_hub",
        "trimesh",
        "networkx",
        "pygltflib",
    )
)


MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU = {
    "L4": gpu.L4(),
    "L40S": gpu.L40S(),
    "A100": gpu.A100(),
    "H100": gpu.H100(),
}.get(GPU_KIND, gpu.L4())


def _volume_path(*parts: str) -> Path:
    return Path(BASE_MOUNT, *parts)


def set_build_vars() -> None:
    os.environ["CCACHE_DIR"] = str(_volume_path("build", "ccache"))
    os.environ["CCACHE_NOHARDLINK"] = "true"
    os.environ["CCACHE_FILE_CLONE"] = "false"
    os.environ["CCACHE_TEMPDIR"] = "/tmp/ccache-tmp"
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    os.environ["CARGO_HOME"] = str(_volume_path("build", "cargo"))
    os.environ["CARGO_TARGET_DIR"] = str(_volume_path("build", "cargo", "target"))
    os.environ["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CUDA_COMPILER_LAUNCHER"] = "ccache"

    directories = [
        os.environ["CCACHE_DIR"],
        os.environ["CARGO_HOME"],
        os.environ["CARGO_TARGET_DIR"],
        os.environ["CCACHE_TEMPDIR"],
        str(_volume_path("workspace")),
        str(_volume_path("logs")),
    ]
    for path in directories:
        Path(path).mkdir(parents=True, exist_ok=True)


@app.function(image=image, gpu=None, volumes={BASE_MOUNT: shared_volume})
def prewarm(model_id: str = MODEL_ID) -> str:
    from huggingface_hub import snapshot_download

    target_dir = _volume_path("models", model_id.replace("/", "__"))
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(target_dir))
    return str(target_dir)


@app.local_entrypoint()
def sync_workspace(src: str = "./splats") -> None:
    volume_name = os.environ.get("MODAL_VOLUME_NAME", DEFAULT_VOLUME_NAME)
    print(f"Syncing {src} -> volume '{volume_name}' mounted at {BASE_MOUNT} ...")
    subprocess.run(["modal", "volume", "put", volume_name, src], check=True)
    print("Done. You can now run build_source or volume_ls.")


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


@app.function(
    image=image,
    gpu=GPU,
    volumes={BASE_MOUNT: shared_volume},
    timeout=30 * 60,
)
def build_source(sm_targets: str = "", multi_sm: bool = False) -> str:
    """Prepare build caches on the shared volume and report selected SM targets."""
    set_build_vars()

    if not sm_targets and multi_sm:
        sm_targets = "7.5;8.0;8.6;8.9;9.0"

    torch_arch, flashinfer_arch = _arch_lists(sm_targets)

    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = flashinfer_arch
    os.environ["UV_CACHE_DIR"] = "/tmp/uvcache"
    Path(os.environ["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    snapshot = {
        "build_root": str(_volume_path("build")),
        "workspace_root": str(_volume_path("workspace")),
        "torch_arch": torch_arch,
        "flashinfer_arch": flashinfer_arch,
    }
    return json.dumps(snapshot, indent=2)


@app.function(image=image, gpu=GPU, volumes={BASE_MOUNT: shared_volume}, timeout=24 * 60 * 60)
def build_gs(repo_ref: str = "main", sm_targets: str = "") -> str:
    set_build_vars()
    torch_arch, _ = _arch_lists(sm_targets)
    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch
    ws_root = _volume_path("workspace")
    repo_dir = build_3dgs(ws_root, ref=repo_ref, torch_arch_list=torch_arch)
    return str(repo_dir)


@app.function(image=image, gpu=None, volumes={BASE_MOUNT: shared_volume}, timeout=2 * 60 * 60)
def init_splats(scene: str, target_count: int = 500000) -> str:
    set_build_vars()
    gltf = _volume_path("scenes", scene, "scene.gltf")
    if not gltf.exists():
        raise FileNotFoundError(str(gltf))
    out_npz = _volume_path("scenes", scene, "init", "splats_init.npz")
    initialize_splats_from_gltf(gltf, target_count, out_npz)
    return str(out_npz)


@app.function(image=image, gpu=None, volumes={BASE_MOUNT: shared_volume}, timeout=60 * 60)
def export_ply(scene: str) -> str:
    set_build_vars()
    in_npz = _volume_path("scenes", scene, "init", "splats_init.npz")
    if not in_npz.exists():
        raise FileNotFoundError(str(in_npz))
    out_ply = _volume_path("scenes", scene, "init", "splats_init.ply")
    export_splats_ply(in_npz, out_ply)
    return str(out_ply)
