#!/usr/bin/env python3
"""Compare U-Ray native and NumPy matrix multiplication on identical inputs."""

import argparse
import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _common import (
    assert_nested_close,
    configure_numpy_threads,
    prepare_backends,
    print_environment,
    run_interleaved,
)

NUMPY_THREADS = configure_numpy_threads()

from tensor import Tensor, set_backend


def _matrix(rows, columns, rng):
    return [
        [rng.uniform(-1.0, 1.0) for _ in range(columns)]
        for _ in range(rows)
    ]


def _run_backend(backend, left, right):
    set_backend(backend)
    return left.matmul(right).data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default="64x64x64,128x784x64,128x64x10,256x256x256,128x768x768",
        help=(
            "comma-separated MxKxN shapes (default includes square matrices "
            "and common dense-layer shapes)"
        ),
    )
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        shapes = [
            tuple(int(dimension) for dimension in value.lower().split("x"))
            for value in args.shapes.split(",")
        ]
    except ValueError:
        parser.error("--shapes must contain comma-separated MxKxN dimensions")
    if (
        not shapes
        or any(len(shape) != 3 or any(size <= 0 for size in shape)
               for shape in shapes)
        or len(set(shapes)) != len(shapes)
        or args.iterations <= 0
        or args.warmups < 0
    ):
        parser.error(
            "shapes must be unique and positive; iterations > 0; warmups >= 0"
        )

    print_environment(
        numpy_threads=NUMPY_THREADS,
        seed=args.seed,
        iterations=args.iterations,
        warmups=args.warmups,
    )
    speedups = []
    try:
        initialization = prepare_backends(
            ("numpy", "native"), set_backend, seed=args.seed
        )
        print(
            "Backend initialization (excluded from timings): "
            + ", ".join(
                f"{backend}={elapsed:.6f}s"
                for backend, elapsed in sorted(initialization.items())
            )
        )
        for m, k, n in shapes:
            rng = random.Random(args.seed + m + k + n)
            left = Tensor(_matrix(m, k, rng))
            right = Tensor(_matrix(k, n, rng))
            result = run_interleaved(
                ("numpy", "native"),
                lambda backend: _run_backend(backend, left, right),
                iterations=args.iterations,
                warmups=args.warmups,
                seed=args.seed,
            )
            numpy_result = result["numpy"]
            native_result = result["native"]
            assert_nested_close(numpy_result["last"], native_result["last"])
            ratio = native_result["median"] / numpy_result["median"]
            speedups.append(1.0 / ratio)
            print(
                f"{m}x{k}x{n}: NumPy={numpy_result['median']:.6f}s "
                f"(MAD {numpy_result['mad']:.6f}s), "
                f"native={native_result['median']:.6f}s "
                f"(MAD {native_result['mad']:.6f}s), "
                f"native/NumPy={ratio:.3f}x"
            )
            print("  Full output comparison: passed.")
    finally:
        set_backend("python")

    geometric_mean = math.exp(
        sum(math.log(value) for value in speedups) / len(speedups)
    )
    print(
        f"Geometric-mean native speed relative to NumPy across these matrix "
        f"sizes: {geometric_mean:.3f}x (below 1 means slower; matmul only)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
