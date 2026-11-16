#!/usr/bin/env python3
"""Compare NumPy and U-Ray native Conv2D forward/backward on MNIST-sized data."""

import argparse
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    return (
        output.data[0][0][0][0],
        x.grad[0][0][0][0],
        w.grad[0][0][0][0],
        b.grad[0],
    )


def benchmark(backend, images, weights, biases, iterations):
    set_backend(backend)
    for _ in range(2):
        probe = run_step(images, weights, biases)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        probe = run_step(images, weights, biases)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0:
        parser.error("--batch-size и --iterations должны быть > 0")

    images = make_values(args.batch_size, 1, 28, 28, 7, 1.0)
    weights = make_values(8, 1, 3, 3, 19, 0.1)
    biases = [index * 0.01 for index in range(8)]

    try:
        numpy_time, numpy_probe = benchmark(
            "numpy", images, weights, biases, args.iterations
        )
    except RuntimeErrorUray as error:
        raise SystemExit(f"Для контрольного замера нужен NumPy: {error}") from error
    try:
        native_time, native_probe = benchmark(
            "native", images, weights, biases, args.iterations
        )
    except RuntimeErrorUray as error:
        raise SystemExit(f"Native backend недоступен: {error}") from error

    for expected, actual in zip(numpy_probe, native_probe):
        if abs(expected - actual) > 1e-8:
            raise SystemExit(
                "Результат native Conv2D отличается от NumPy больше чем на 1e-8"
            )

    print(f"batch={args.batch_size}, input=1x28x28, filters=8x3x3")
    print(f"NumPy forward+backward median: {numpy_time:.6f}s")
    print(f"native forward+backward median: {native_time:.6f}s")
    print(f"native / NumPy: {native_time / numpy_time:.3f}x")
    print("Output and gradient probes match.")


if __name__ == "__main__":
    main()
