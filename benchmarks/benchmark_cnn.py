#!/usr/bin/env python3
"""Compare full U-Ray CNN training/evaluation with NumPy and native backends."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errors import RuntimeErrorUray
from mnist import MNISTView
from modules import ModuleRegistry
from tensor import set_backend


class PrefixMNISTView(MNISTView):
    """Bound a lazy MNIST view to its first N records for quick comparisons."""

    __slots__ = ("limit",)

    def __init__(self, source, limit):
        super().__init__(source.storage, source.kind)
        self.limit = limit

    def __len__(self):
        return self.limit


def _build_model(nn):
    members = nn.members
    return members["sequential"].fn([[
        members["conv2d_layer"].fn([1, 8, 3, 1, 1]),
        members["activation_layer"].fn(["relu"]),
        members["max_pool2d_layer"].fn([2, 2]),
        members["flatten_layer"].fn([]),
        members["linear_layer"].fn([8 * 14 * 14, 10]),
    ]])


def _bounded_split(split, limit):
    return {
        "X": PrefixMNISTView(split["X"], limit),
        "Y": PrefixMNISTView(split["Y"], limit),
    }


def _run_backend(nn, dataset, backend, epochs, batch_size):
    set_backend(backend)
    nn.members["rand_seed"].fn([42])
    model = _build_model(nn)
    nn.members["rand_seed"].fn([2026])

    train = dataset["train"]
    test = dataset["test"]
    started = time.perf_counter()
    history = nn.members["train_layers"].fn([
        model, train["X"], train["Y"], epochs, 0.001, batch_size,
        "cce", "adam", False, [1, 28, 28],
    ])
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    metrics = nn.members["evaluate_layers"].fn([
        model, test["X"], test["Y"], "cce", [1, 28, 28],
    ])
    evaluate_seconds = time.perf_counter() - started
    return {
        "history": history,
        "metrics": metrics,
        "train_seconds": train_seconds,
        "evaluate_seconds": evaluate_seconds,
        "total_seconds": train_seconds + evaluate_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "examples" / "data" / "mnist"),
        help="каталог с MNIST IDX-архивами",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="сравнить 60000 train и 10000 test изображений",
    )
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-slowdown-seconds",
        type=float,
        default=5.0,
        help="максимальное допустимое отставание native от NumPy",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="минимальная точность на тестовой выборке",
    )
    args = parser.parse_args()
    if min(args.train_samples, args.test_samples, args.epochs,
           args.batch_size) <= 0 or args.max_slowdown_seconds < 0 or not (
               0.0 <= args.min_accuracy <= 1.0
           ):
        parser.error(
            "sizes, epochs, batch-size должны быть > 0, "
            "max-slowdown-seconds должен быть >= 0, "
            "min-accuracy должен быть в [0, 1]"
        )

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error(
            f"не найден каталог {data_dir}; сначала запустите "
            "python3 examples/download_mnist.py"
        )

    nn = ModuleRegistry().get("nn")
    try:
        loaded = nn.members["load_mnist"].fn([str(data_dir), False])
    except RuntimeErrorUray as error:
        parser.error(str(error))
    train_count = len(loaded["train"]["X"])
    test_count = len(loaded["test"]["X"])
    if train_count != 60000 or test_count != 10000:
        parser.error("для бенчмарка нужны полные 60000 train / 10000 test")

    if args.full:
        train_samples, test_samples = train_count, test_count
    else:
        train_samples = args.train_samples
        test_samples = args.test_samples
    if train_samples > train_count or test_samples > test_count:
        parser.error("выбранный размер превышает полный MNIST split")
    dataset = {
        "train": _bounded_split(loaded["train"], train_samples),
        "test": _bounded_split(loaded["test"], test_samples),
    }

    print("Running NumPy CNN train/evaluate...", flush=True)
    try:
        numpy_result = _run_backend(
            nn, dataset, "numpy", args.epochs, args.batch_size
        )
    except RuntimeErrorUray as error:
        print(
            f"NumPy baseline недоступен: {error}\n"
            "Установите NumPy только для контрольного замера.",
            file=sys.stderr,
        )
        return 2

    print("Running native CNN train/evaluate...", flush=True)
    try:
        native_result = _run_backend(
            nn, dataset, "native", args.epochs, args.batch_size
        )
    except RuntimeErrorUray as error:
        print(f"Native backend недоступен: {error}", file=sys.stderr)
        return 2
    finally:
        set_backend("python")

    ratio = native_result["total_seconds"] / numpy_result["total_seconds"]
    difference = (
        native_result["total_seconds"] - numpy_result["total_seconds"]
    )
    slowdown = max(0.0, difference)
    print("MNIST CNN: Conv2D(1, 8, 3) -> ReLU -> MaxPool(2) -> Dense(1568, 10)")
    print(
        f"  samples: train={train_samples}, test={test_samples}, "
        f"epochs={args.epochs}, batch={args.batch_size}"
    )
    print(
        f"  NumPy:  train={numpy_result['train_seconds']:.3f}s, "
        f"evaluate={numpy_result['evaluate_seconds']:.3f}s, "
        f"total={numpy_result['total_seconds']:.3f}s"
    )
    print(
        f"  native: train={native_result['train_seconds']:.3f}s, "
        f"evaluate={native_result['evaluate_seconds']:.3f}s, "
        f"total={native_result['total_seconds']:.3f}s"
    )
    print(
        f"  native/NumPy={ratio:.3f}x "
        f"(native {'slower' if difference > 0 else 'faster'} by "
        f"{abs(difference):.3f}s; max allowed slowdown "
        f"{args.max_slowdown_seconds:.3f}s)"
    )
    print(
        f"  test accuracy: NumPy={numpy_result['metrics'][1]:.5f}, "
        f"native={native_result['metrics'][1]:.5f}"
    )
    accuracy_passed = min(
        numpy_result["metrics"][1], native_result["metrics"][1]
    ) >= args.min_accuracy
    if not accuracy_passed:
        print(
            f"  QUALITY: FAIL (minimum accuracy "
            f"{args.min_accuracy:.2%} not reached)"
        )
    if slowdown <= args.max_slowdown_seconds and accuracy_passed:
        print("  RESULT: PASS")
        return 0
    print("  RESULT: NOT MET")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
