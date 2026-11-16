#!/usr/bin/env python3
"""Compare U-Ray native and NumPy Transformer train/evaluate throughput."""

import argparse
import random
import statistics
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errors import RuntimeErrorUray
from modules import ModuleRegistry
from tensor import set_backend


def _make_data(samples, sequence, features, seed):
    rng = random.Random(seed)
    inputs = [
        [
            [rng.uniform(-1.0, 1.0) for _ in range(features)]
            for _ in range(sequence)
        ]
        for _ in range(samples)
    ]
    targets = [
        [[value * 0.5 for value in row] for row in sample]
        for sample in inputs
    ]
    return inputs, targets


def _run(nn, backend, inputs, targets, args):
    set_backend(backend)
    nn.members["rand_seed"].fn([42])
    model = nn.members["sequential"].fn([[
        nn.members["transformer_block"].fn([
            args.features, args.heads, args.hidden_dim, 0.0, True,
        ]),
        nn.members["linear_layer"].fn([args.features, args.features]),
    ]])
    nn.members["rand_seed"].fn([2026])

    started = time.perf_counter()
    nn.members["train_layers"].fn([
        model, inputs, targets, args.epochs, 0.001, args.batch_size,
        "mse", "adam", False,
    ])
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    metrics = nn.members["evaluate_layers"].fn([
        model, inputs[:args.eval_samples], targets[:args.eval_samples], "mse",
    ])
    evaluate_seconds = time.perf_counter() - started
    return train_seconds, evaluate_seconds, metrics[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.5,
        help="maximum accepted native/NumPy total-time ratio",
    )
    args = parser.parse_args()
    if (
        min(
            args.samples, args.eval_samples, args.sequence_length,
            args.features, args.heads, args.hidden_dim, args.batch_size,
            args.epochs, args.iterations,
        ) <= 0
        or args.eval_samples > args.samples
        or args.features % args.heads
        or args.max_ratio <= 0
    ):
        parser.error("invalid dimensions, counts, or max-ratio")

    inputs, targets = _make_data(
        args.samples, args.sequence_length, args.features, 314159
    )
    nn = ModuleRegistry().get("nn")
    results = {}
    try:
        for backend in ("numpy", "native"):
            try:
                _run(nn, backend, inputs, targets, args)  # warmup
                samples = [
                    _run(nn, backend, inputs, targets, args)
                    for _ in range(args.iterations)
                ]
            except RuntimeErrorUray as error:
                parser.error(f"{backend} backend недоступен: {error}")
            train_seconds = statistics.median(
                result[0] for result in samples
            )
            evaluate_seconds = statistics.median(
                result[1] for result in samples
            )
            results[backend] = (
                train_seconds, evaluate_seconds, samples[-1][2]
            )
    finally:
        set_backend("python")

    numpy_result = results["numpy"]
    native_result = results["native"]
    numpy_total = sum(numpy_result[:2])
    native_total = sum(native_result[:2])
    ratio = native_total / numpy_total
    print(
        f"Transformer: samples={args.samples}, eval={args.eval_samples}, "
        f"sequence={args.sequence_length}, features={args.features}, "
        f"heads={args.heads}, batch={args.batch_size}, epochs={args.epochs}, "
        f"median of {args.iterations} runs after warmup"
    )
    print(
        f"  NumPy:  train={numpy_result[0]:.4f}s, "
        f"evaluate={numpy_result[1]:.4f}s, total={numpy_total:.4f}s"
    )
    print(
        f"  native: train={native_result[0]:.4f}s, "
        f"evaluate={native_result[1]:.4f}s, total={native_total:.4f}s"
    )
    print(
        f"  native/NumPy={ratio:.3f}x; "
        f"evaluation loss NumPy={numpy_result[2]:.8g}, "
        f"native={native_result[2]:.8g}"
    )
    if ratio <= args.max_ratio:
        print("  RESULT: PASS")
        return 0
    print(f"  RESULT: NOT MET (maximum ratio {args.max_ratio:.3f}x)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
