#!/usr/bin/env python3
"""Compare a full MLP MNIST train/evaluate cycle with NumPy on this CPU."""

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errors import RuntimeErrorUray
from modules import ModuleRegistry
from tensor import set_backend


def _run_backend(nn, dataset, backend, epochs, batch_size):
    set_backend(backend)
    nn.members["rand_seed"].fn([42])
    model = nn.members["mlp"].fn([[784, 64, 10], "relu", "softmax"])
    nn.members["rand_seed"].fn([2026])

    train = dataset["train"]
    test = dataset["test"]
    started = time.perf_counter()
    history = nn.members["train"].fn([
        model, train["X"], train["Y"], epochs, 0.001, batch_size,
        "cce", "adam", False,
    ])
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    metrics = nn.members["evaluate"].fn([
        model, test["X"], test["Y"], "cce",
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-ratio", type=float, default=1.5)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="минимальная точность на тестовой выборке",
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0 or args.batch_size <= 0 or args.max_ratio <= 0
        or not 0.0 <= args.min_accuracy <= 1.0
    ):
        parser.error(
            "epochs, batch-size и max-ratio должны быть > 0; "
            "min-accuracy должен быть в [0, 1]"
        )

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error(
            f"не найден каталог {data_dir}; сначала запустите "
            "python3 examples/download_mnist.py"
        )

    nn = ModuleRegistry().get("nn")
    dataset = nn.members["load_mnist"].fn([str(data_dir), False])
    if len(dataset["train"]["X"]) != 60000 or len(dataset["test"]["X"]) != 10000:
        parser.error("для бенчмарка нужны полные 60000 train / 10000 test")

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

    try:
        native_result = _run_backend(
            nn, dataset, "native", args.epochs, args.batch_size
        )
    except RuntimeErrorUray as error:
        print(f"Нативный backend недоступен: {error}", file=sys.stderr)
        return 2
    finally:
        set_backend("python")

    ratio = native_result["total_seconds"] / numpy_result["total_seconds"]
    print("MNIST 784-64-10, Adam, CCE")
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
        f"(target <= {args.max_ratio:.3f}x)"
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
    if ratio <= args.max_ratio and accuracy_passed:
        print("  RESULT: PASS")
        return 0
    print("  RESULT: NOT MET")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
