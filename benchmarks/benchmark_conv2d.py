#!/usr/bin/env python3
"""Compare NumPy and U-Ray native Conv2D forward/backward on MNIST-sized data."""

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _common import (
    assert_nested_close,
    configure_numpy_threads,
    print_environment,
    prepare_backends,
    run_interleaved,
)

NUMPY_THREADS = configure_numpy_threads()

from autograd import GradTensor
from errors import RuntimeErrorUray
from tensor import set_backend


def make_values(batch_size, channels, height, width, seed, scale):
    rng = random.Random(seed)
    return [
        [
            [
                [rng.uniform(-scale, scale) for _ in range(width)]
                for _ in range(height)
            ]
            for _ in range(channels)
        ]
        for _ in range(batch_size)
    ]


def run_step(images, weights, biases):
    x = GradTensor(images, requires_grad=True)
    w = GradTensor(weights, requires_grad=True)
    b = GradTensor(biases, requires_grad=True)
    output = x.conv2d(w, b, stride=1, padding=1)
    output.mean().backward()
    return output.data, x.grad, w.grad, b.grad


def run_backend(backend, images, weights, biases):
    set_backend(backend)
    return run_step(images, weights, biases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0 or args.warmups < 0:
        parser.error("batch-size и iterations должны быть > 0; warmups >= 0")

    images = make_values(args.batch_size, 1, 28, 28, 7, 1.0)
    weights = make_values(8, 1, 3, 3, 19, 0.1)
    biases = [index * 0.01 for index in range(8)]

    try:
        initialization = prepare_backends(
            ("numpy", "native"), set_backend, seed=args.seed
        )
        result = run_interleaved(
            ("numpy", "native"),
            lambda backend: run_backend(backend, images, weights, biases),
            iterations=args.iterations,
            warmups=args.warmups,
            seed=args.seed,
        )
    except RuntimeErrorUray as error:
        raise SystemExit(f"Не удалось выполнить оба backend: {error}") from error
    finally:
        set_backend("python")

    numpy_result = result["numpy"]
    native_result = result["native"]
    try:
        assert_nested_close(numpy_result["last"], native_result["last"])
    except AssertionError as error:
        raise SystemExit(f"Результат/градиенты Conv2D не совпали: {error}") from error

    print_environment(
        numpy_threads=NUMPY_THREADS,
        seed=args.seed,
        iterations=args.iterations,
        warmups=args.warmups,
    )
    print(
        "Backend initialization (excluded from timing): "
        f"NumPy={initialization['numpy']:.3f}s, "
        f"native={initialization['native']:.3f}s"
    )

    print(f"batch={args.batch_size}, input=1x28x28, filters=8x3x3")
    print(
        f"NumPy forward+backward median: {numpy_result['median']:.6f}s "
        f"(MAD {numpy_result['mad']:.6f}s)"
    )
    print(
        f"native forward+backward median: {native_result['median']:.6f}s "
        f"(MAD {native_result['mad']:.6f}s)"
    )
    print(f"native / NumPy: {native_result['median'] / numpy_result['median']:.3f}x")
    print("Full outputs and gradients match.")


if __name__ == "__main__":
    main()
