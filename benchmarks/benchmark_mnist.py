#!/usr/bin/env python3
"""Compare a full MLP MNIST train/evaluate cycle with NumPy on this CPU."""

import argparse
from pathlib import Path
import sys
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _common import (
    configure_numpy_threads,
    prepare_backends,
    print_environment,
    run_interleaved,
)

NUMPY_THREADS = configure_numpy_threads()

from errors import RuntimeErrorUray
from mnist import MNISTView
from modules import ModuleRegistry
from tensor import set_backend


class PrefixMNISTView(MNISTView):
    """Bound a lazy MNIST view to its first N records."""

    __slots__ = ("limit",)

    def __init__(self, source, limit):
        super().__init__(source.storage, source.kind)
        self.limit = limit

    def __len__(self):
        return self.limit


def _bounded_split(split, limit):
    return {
        "X": PrefixMNISTView(split["X"], limit),
        "Y": PrefixMNISTView(split["Y"], limit),
    }


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
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-ratio", type=float, default=1.5)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="минимальная точность на тестовой выборке",
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0 or args.batch_size <= 0
        or args.iterations <= 0 or args.warmups < 0
        or args.max_ratio <= 0
        or (args.train_samples is not None and args.train_samples <= 0)
        or (args.test_samples is not None and args.test_samples <= 0)
        or not 0.0 <= args.min_accuracy <= 1.0
    ):
        parser.error(
            "epochs, batch-size, iterations и max-ratio должны быть > 0; "
            "warmups должен быть >= 0; "
            "min-accuracy должен быть в [0, 1]"
        )

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        parser.error(
            f"не найден каталог {data_dir}; сначала запустите "
            "python3 examples/download_mnist.py"
        )

    nn = ModuleRegistry().get("nn")
    loaded = nn.members["load_mnist"].fn([str(data_dir), False])
    train_count = len(loaded["train"]["X"])
    test_count = len(loaded["test"]["X"])
    if train_count != 60000 or test_count != 10000:
        parser.error("для бенчмарка нужны полные 60000 train / 10000 test")
    train_samples = args.train_samples or train_count
    test_samples = args.test_samples or test_count
    if train_samples > train_count or test_samples > test_count:
        parser.error("выбранный размер превышает полный MNIST split")
    dataset = {
        "train": _bounded_split(loaded["train"], train_samples),
        "test": _bounded_split(loaded["test"], test_samples),
    }

    try:
        initialization = prepare_backends(
            ("numpy", "native"), set_backend, seed=args.seed
        )
        results = run_interleaved(
            ("numpy", "native"),
            lambda backend: _run_backend(
                nn, dataset, backend, args.epochs, args.batch_size
            ),
            iterations=args.iterations,
            warmups=args.warmups,
            seed=args.seed,
            keep_results=True,
        )
    except RuntimeErrorUray as error:
        print(
            f"Не удалось выполнить оба backend: {error}", file=sys.stderr
        )
        return 2
    finally:
        set_backend("python")

    def summarize(backend):
        summary = results[backend]
        runs = summary["results"]
        train_seconds = statistics.median(
            result["train_seconds"] for result in runs
        )
        evaluate_seconds = statistics.median(
            result["evaluate_seconds"] for result in runs
        )
        return {
            "train_seconds": train_seconds,
            "evaluate_seconds": evaluate_seconds,
            "total_seconds": train_seconds + evaluate_seconds,
            "accuracy": statistics.median(
                result["metrics"][1] for result in runs
            ),
        }

    numpy_result = summarize("numpy")
    native_result = summarize("native")
    ratio = native_result["total_seconds"] / numpy_result["total_seconds"]
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
    print(
        f"MNIST MLP 784-64-10, Adam, CCE; train={train_samples}, "
        f"test={test_samples}, epochs={args.epochs}"
    )
    print(
        f"  NumPy:  train={numpy_result['train_seconds']:.3f}s, "
        f"evaluate={numpy_result['evaluate_seconds']:.3f}s, "
        f"total={numpy_result['total_seconds']:.3f}s (median)"
    )
    print(
        f"  native: train={native_result['train_seconds']:.3f}s, "
        f"evaluate={native_result['evaluate_seconds']:.3f}s, "
        f"total={native_result['total_seconds']:.3f}s (median)"
    )
    print(
        f"  paired trial wall: NumPy={results['numpy']['median']:.3f}s "
        f"(MAD {results['numpy']['mad']:.3f}s), "
        f"native={results['native']['median']:.3f}s "
        f"(MAD {results['native']['mad']:.3f}s)"
    )
    print(
        f"  native/NumPy={ratio:.3f}x "
        f"(target <= {args.max_ratio:.3f}x)"
    )
    print(
        f"  test accuracy: NumPy={numpy_result['accuracy']:.5f}, "
        f"native={native_result['accuracy']:.5f}"
    )
    accuracy_passed = min(
        numpy_result["accuracy"], native_result["accuracy"]
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
