#!/usr/bin/env python3
"""Validate generated preview PNGs are not uniformly black."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import imageio.v2 as iio
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "check_previews requires numpy and imageio. Activate the venv before running."
    ) from exc


def _is_valid(path: Path, min_value: float) -> bool:
    image = iio.imread(path)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return float(np.max(arr)) > min_value


def _check_directory(root: Path, min_value: float) -> list[Path]:
    failures: list[Path] = []
    for path in sorted(root.rglob("*.png")):
        if not _is_valid(path, min_value):
            failures.append(path)
    return failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Preview output directory to validate")
    parser.add_argument(
        "--min-value",
        type=float,
        default=1.0,
        help="Minimum max pixel value required for success (default: 1.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    directory = args.directory.resolve()
    if not directory.exists():
        print(f"Directory not found: {directory}", file=sys.stderr)
        return 2

    failures = _check_directory(directory, args.min_value)
    if failures:
        print("Detected black or near-black previews:", file=sys.stderr)
        for path in failures:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(f"All previews in {directory} passed the brightness check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
