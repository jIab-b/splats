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
        dirs_to_sync = ['dreamgaussian']
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
    gpu="A100",
    volumes={"/workspace": splats_wspace},
)
def install_all():
    import sys, os, subprocess, pathlib
    sys.path.insert(0, "/workspace/3dgs")

    venv_path = "/workspace/venv"
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

    os.chdir("/workspace/dreamgaussian")
    subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], env=venv_env, check=True)

    os.chdir("/workspace/dreamgaussian/diff-gaussian-rasterization")
    subprocess.run(["uv", "pip", "install", "wheel"], env=venv_env, check=True)
    subprocess.run(["rm", "-rf", "build"], check=False)

    build_env = venv_env.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "8.0"


    subprocess.run([
        "uv", "pip", "install", "--python", f"{venv_path}/bin/python",
        "--no-build-isolation", "."
    ], env=build_env, cwd="/workspace/dreamgaussian/diff-gaussian-rasterization", check=True)

    # Install additional deps
    if os.path.exists("/workspace/dreamgaussian/simple-knn"):
        try:
            subprocess.run([
                "uv", "pip", "install", "--python", f"{venv_path}/bin/python",
                "--no-build-isolation", "."
            ], env=build_env, cwd="/workspace/dreamgaussian/simple-knn", check=True)
        except Exception:
            print("simple-knn not found locally; skipping local install")

    subprocess.run(["uv", "pip", "install", "git+https://github.com/NVlabs/nvdiffrast/"], env=venv_env, check=True)
    subprocess.run(["uv", "pip", "install", "git+https://github.com/ashawkey/kiuikit"], env=venv_env, check=True)

    splats_wspace.commit()



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
    timeout=1800,  # 30 minutes
)
def inference(prompt: str = "", output_name: str = "result"):
    import sys, os, glob, site
    venv_path = "/workspace/venv"
    venv_env = os.environ.copy()
    venv_env["PATH"] = f"{venv_path}/bin:" + venv_env.get("PATH", "")
    venv_env["VIRTUAL_ENV"] = venv_path
    os.environ["PATH"] = venv_env["PATH"]
    os.environ["VIRTUAL_ENV"] = venv_path

    # Properly activate virtual environment
    try:
        venv_site = sorted(glob.glob(f"{venv_path}/lib/python*/site-packages"))[-1]
        site.addsitedir(venv_site)
    except Exception:
        pass
    try:
        conda_site = sorted(glob.glob("/opt/conda/lib/python*/site-packages"))[-1]
        site.addsitedir(conda_site)
    except Exception:
        pass
    try:
        for p in glob.glob("/workspace/diff-gaussian-rasterization/build/lib.*"):
            if p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass

    input_image_path = "/workspace/images/telvanni.jpg"
    print(f"Starting DreamGaussian inference for: {input_image_path}")
    
    # Stage 1: Generate Gaussian Splatting model
    print("=== Stage 1: Gaussian Splatting ===")
    sys.path.insert(0, "/workspace/dreamgaussian")
    
    import subprocess
    import yaml
    
    # Create config for stage 1
    config = {
        'input': input_image_path,
        'prompt': prompt,
        'negative_prompt': '',
        'elevation': 0,
        'ref_size': 256,
        'density_thresh': 1,
        'outdir': '/workspace/out_local',
        'mesh_format': 'obj',
        'save_path': output_name,
        'mvdream': False,
        'imagedream': False,
        'stable_zero123': False,
        'lambda_sd': 0,
        'lambda_zero123': 1,
        'warmup_rgb_loss': True,
        'batch_size': 1,
        'iters': 500,
        'anneal_timestep': True,
        'iters_refine': 50,
        'radius': 2,
        'fovy': 49.1,
        'min_ver': -30,
        'max_ver': 30,
        'load': None,
        'train_geo': False,
        'invert_bg_prob': 0.5,
        'gui': False,
        'force_cuda_rast': False,
        'H': 800,
        'W': 800,
        'num_pts': 5000,
        'sh_degree': 0,
        'position_lr_init': 0.001,
        'position_lr_final': 0.00002,
        'position_lr_delay_mult': 0.02,
        'position_lr_max_steps': 500,
        'feature_lr': 0.01,
        'opacity_lr': 0.05,
        'scaling_lr': 0.005,
        'rotation_lr': 0.005,
        'percent_dense': 0.01,
        'density_start_iter': 100,
        'density_end_iter': 3000,
        'densification_interval': 100,
        'opacity_reset_interval': 700,
        'densify_grad_threshold': 0.01,
        'geom_lr': 0.0001,
        'texture_lr': 0.2
    }
    
    # Save config
    config_path = f"/workspace/{output_name}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Use venv Python explicitly
    python_exe = f"{venv_path}/bin/python"

    # Run stage 1
    cmd1 = [python_exe, "main.py", "--config", config_path]
    print(f"Running: {' '.join(cmd1)}")
    result1 = subprocess.run(cmd1, cwd="/workspace/dreamgaussian", check=True, env=venv_env)

    # Stage 2: Extract textured mesh
    print("\n=== Stage 2: Mesh Extraction ===")

    # Update config for stage 2
    config['mesh'] = f"/workspace/out_local/{output_name}_mesh.obj"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Run stage 2
    cmd2 = [python_exe, "main2.py", "--config", config_path]
    print(f"Running: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, cwd="/workspace/dreamgaussian", check=True, env=venv_env)

    # Generate video
    print("\n=== Rendering Video ===")
    final_mesh = f"/workspace/out_local/{output_name}.obj"
    video_path = f"/workspace/out_local/{output_name}.mp4"

    if os.path.exists(final_mesh):
        cmd3 = [python_exe, "-m", "kiui.render", final_mesh, "--save_video", video_path, "--wogui"]
        print(f"Running: {' '.join(cmd3)}")
        result3 = subprocess.run(cmd3, cwd="/workspace/dreamgaussian", check=False, env=venv_env)  # Don't fail if video fails
    else:
        print(f"Warning: Mesh file {final_mesh} not found")
    
    splats_wspace.commit()
    
    # Return paths to generated files
    output_files = {
        'gaussian_model': f"/workspace/out_local/{output_name}_model.ply",
        'mesh': final_mesh,
        'video': video_path if os.path.exists(video_path) else None,
        'config': config_path
    }
    
    print(f"Inference complete! Outputs in: {output_files}")
    return output_files


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def test_install():
    import sys, os, glob, site

    venv_path = "/workspace/venv"
    venv_env = os.environ.copy()
    venv_env["PATH"] = f"{venv_path}/bin:" + venv_env.get("PATH", "")
    venv_env["VIRTUAL_ENV"] = venv_path
    os.environ["PATH"] = venv_env["PATH"]
    os.environ["VIRTUAL_ENV"] = venv_path

    # Properly activate virtual environment
    try:
        venv_site = sorted(glob.glob(f"{venv_path}/lib/python*/site-packages"))[-1]
        site.addsitedir(venv_site)
    except Exception:
        pass
    try:
        conda_site = sorted(glob.glob("/opt/conda/lib/python*/site-packages"))[-1]
        site.addsitedir(conda_site)
    except Exception:
        pass
    try:
        for p in glob.glob("/workspace/diff-gaussian-rasterization/build/lib.*"):
            if p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass

    print("Testing dependency installations...")

    # Test PyTorch and basic ML deps
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")

    # Test diff-gaussian-rasterization
    try:
        import diff_gaussian_rasterization
        print("✓ diff_gaussian_rasterization installed")
    except ImportError as e:
        print(f"✗ diff_gaussian_rasterization import failed: {e}")

    # Test simple-knn if available
    try:
        import simple_knn
        print("✓ simple_knn installed")
    except ImportError:
        print("⚠ simple_knn not available (skipped)")

    # Test nvdiffrast
    try:
        import nvdiffrast.torch as dr
        print("✓ nvdiffrast installed")
    except ImportError as e:
        print(f"✗ nvdiffrast import failed: {e}")

    # Test kiui (from kiuikit)
    try:
        import kiui
        print("✓ kiui (kiuikit) installed")
    except ImportError as e:
        print(f"✗ kiui import failed: {e}")

    # Test basic functionality
    try:
        import numpy as np
        import torch

        # Simple tensor operation to verify CUDA if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.t())
        print(f"✓ Basic tensor operations work on {device}")

    except Exception as e:
        print(f"✗ Basic tensor operations failed: {e}")

    print("Dependency test complete!")


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def shell():
    import sys
    import subprocess
    subprocess.call(["/bin/bash"], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
