#!/usr/bin/env python3
"""Create per-project virtualenvs with uv and verify imports."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UV = os.environ.get("UV_BIN", "uv")
PY_SUFFIX = "Scripts\\python.exe" if os.name == "nt" else "bin/python"
TORCH_ARGS = [
    "torch==2.7.0",
    "torchvision",
    "--index-url",
    "https://download.pytorch.org/whl/cu126",
]

ENV_SPECS = [
    {
        "name": "venv_gsplat",
        "commands": [
            TORCH_ARGS,
            ["--no-build-isolation", "-e", str(BASE_DIR / "gsplat")],
        ],
        "checks": [
            "import gsplat; print('gsplat ok')",
        ],
    },
    {
        "name": "venv_nerfstudio",
        "commands": [
            TORCH_ARGS,
            ["--no-build-isolation", "-e", str(BASE_DIR / "gsplat")],
            ["--no-build-isolation", "-e", str(BASE_DIR / "nerfstudio")],
        ],
        "checks": [
            "from nerfstudio.models import splatfacto; print('splatfacto model', splatfacto.SplatfactoModel.__name__)",
        ],
    },
    {
        "name": "venv_dreamgaussian",
        "commands": [
            TORCH_ARGS,
            ["-r", str(BASE_DIR / "dreamgaussian" / "requirements.txt")],
            ["--no-build-isolation", "-e", str(BASE_DIR / "dreamgaussian" / "simple-knn")],
            ["--no-build-isolation", "-e", str(BASE_DIR / "dreamgaussian" / "diff-gaussian-rasterization")],
            ["git+https://github.com/ashawkey/kiuikit"],
        ],
        "checks": [
            "import diff_gaussian_rasterization as dgr; from simple_knn import distCUDA2; print('dreamgaussian deps ok')",
        ],
    },
    {
        "name": "venv_stable_dreamfusion",
        "commands": [
            TORCH_ARGS,
            ["-r", str(BASE_DIR / "stable-dreamfusion" / "requirements.txt")],
        ],
        "checks": [
            "import sys, pathlib; sys.path.append(str(pathlib.Path('stable-dreamfusion'))); import main as sdf_main; print('stable-dreamfusion ok')",
        ],
    },
    {
        "name": "venv_dust3r",
        "commands": [
            TORCH_ARGS,
            ["-r", str(BASE_DIR / "dust3r" / "requirements.txt")],
        ],
        "checks": [
            "import dust3r; print('dust3r ok')",
        ],
    },
    {
        "name": "venv_depthanything",
        "commands": [
            TORCH_ARGS,
            ["-r", str(BASE_DIR / "Depth-Anything" / "requirements.txt")],
        ],
        "checks": [
            "import depth_anything; print('depth-anything ok')",
        ],
    },
]


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_venv(path: Path) -> Path:
    if not path.exists():
        run([UV, "venv", str(path)])
    python_bin = path / PY_SUFFIX
    if not python_bin.exists():
        raise RuntimeError(f"Python executable not found in {path}")
    return python_bin


def install_with_uv(python: Path, args: list[str]) -> None:
    run([UV, "pip", "install", "--python", str(python), *args])


def run_checks(python: Path, snippets: list[str]) -> None:
    for snippet in snippets:
        run([str(python), "-c", snippet])


def main() -> None:
    for spec in ENV_SPECS:
        path = BASE_DIR / spec["name"]
        print(f"\n=== Setting up {path.name} ===")
        python = ensure_venv(path)
        for args in spec["commands"]:
            install_with_uv(python, list(args))
        run_checks(python, list(spec.get("checks", [])))
        print(f"=== {path.name} ready ===\n")


if __name__ == "__main__":
    main()
